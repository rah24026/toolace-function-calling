import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp


def parse_sse_line(line):
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return "[DONE]"
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def load_prompts(data_dir, n):
    from datasets import load_from_disk

    ds = load_from_disk(data_dir)["validation"]
    n = min(n, len(ds))
    prompts = []
    for ex in ds.select(range(n)):
        messages = []
        for m in ex["messages"]:
            if m["role"] == "assistant":
                break
            messages.append(m)
        if not messages or messages[-1]["role"] != "user":
            continue
        prompts.append({"messages": messages, "tools": ex["tools"] or None})
    return prompts


async def send_one(session, base_url, model, prompt):
    payload = {
        "model": model,
        "messages": prompt["messages"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "max_tokens": 512,
    }
    if prompt.get("tools"):
        payload["tools"] = prompt["tools"]
        payload["tool_choice"] = "auto"

    t_start = time.perf_counter()
    t_first_token = None
    num_output_tokens = None

    async with session.post(f"{base_url}/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="ignore")
            event = parse_sse_line(line)
            if event is None or event == "[DONE]":
                continue
            usage = event.get("usage")
            if usage:
                num_output_tokens = usage.get("completion_tokens")
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            has_content = delta.get("content") or delta.get("tool_calls")
            if has_content and t_first_token is None:
                t_first_token = time.perf_counter()

    t_end = time.perf_counter()
    ttft = (t_first_token - t_start) if t_first_token else None
    e2e = t_end - t_start
    tpot = None
    if ttft is not None and num_output_tokens and num_output_tokens > 1:
        tpot = (e2e - ttft) / (num_output_tokens - 1)

    return {"ttft_s": ttft, "e2e_s": e2e, "tpot_s": tpot, "output_tokens": num_output_tokens}


async def run_at_concurrency(base_url, model, prompts, concurrency, num_requests):
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def worker(idx):
        prompt = prompts[idx % len(prompts)]
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                r = await send_one(session, base_url, model, prompt)
                results.append(r)

    wall_start = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in range(num_requests)))
    wall_time = time.perf_counter() - wall_start

    ttfts = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    e2es = [r["e2e_s"] for r in results if r["e2e_s"] is not None]
    tpots = [r["tpot_s"] for r in results if r["tpot_s"] is not None]
    total_output_tokens = sum(r["output_tokens"] or 0 for r in results)

    def pct(values, p):
        if not values:
            return None
        s = sorted(values)
        k = int(round((p / 100) * (len(s) - 1)))
        return s[k]

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "wall_time_s": wall_time,
        "request_throughput_rps": num_requests / wall_time if wall_time > 0 else None,
        "output_token_throughput_tok_s": total_output_tokens / wall_time if wall_time > 0 else None,
        "ttft_s": {"mean": statistics.mean(ttfts) if ttfts else None, "p50": pct(ttfts, 50), "p90": pct(ttfts, 90), "p99": pct(ttfts, 99)},
        "e2e_latency_s": {"mean": statistics.mean(e2es) if e2es else None, "p50": pct(e2es, 50), "p90": pct(e2es, 90), "p99": pct(e2es, 99)},
        "tpot_s": {"mean": statistics.mean(tpots) if tpots else None, "p50": pct(tpots, 50), "p90": pct(tpots, 90), "p99": pct(tpots, 99)},
    }


async def main_async(args):
    prompts = load_prompts(args.data_dir, args.num_prompt_pool)

    all_results = []
    for concurrency in args.concurrency_levels:
        result = await run_at_concurrency(args.base_url, args.model, prompts, concurrency, args.num_requests)
        all_results.append(result)
        print(json.dumps(result, indent=2))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="qwen25-7b-toolace")
    parser.add_argument("--data_dir", default="data/toolace_processed")
    parser.add_argument("--concurrency_levels", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--num_requests", type=int, default=200)
    parser.add_argument("--num_prompt_pool", type=int, default=300)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
