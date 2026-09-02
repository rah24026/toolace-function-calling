import argparse
import ast
import json
import re
from pathlib import Path


def _find_matching(s, start, open_ch, close_ch):
    depth = 0
    i = start
    while i < len(s):
        if s[i] == open_ch:
            depth += 1
        elif s[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def split_top_level(s, sep=","):
    parts = []
    depth = 0
    in_str = None
    buf = []
    i = 0
    while i < len(s):
        c = s[i]
        if in_str:
            buf.append(c)
            if c == "\\":
                i += 1
                if i < len(s):
                    buf.append(s[i])
            elif c == in_str:
                in_str = None
            i += 1
            continue
        if c in ("'", '"'):
            in_str = c
            buf.append(c)
        elif c in "([{":
            depth += 1
            buf.append(c)
        elif c in ")]}":
            depth -= 1
            buf.append(c)
        elif c == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    if buf:
        parts.append("".join(buf))
    return [p for p in (p.strip() for p in parts) if p]


def split_calls(s):
    calls = []
    n = len(s)
    i = 0
    start = 0
    depth = 0
    started = False
    while i < n:
        c = s[i]
        if c in ("'", '"'):
            quote = c
            i += 1
            while i < n and s[i] != quote:
                i += 2 if s[i] == "\\" else 1
            i += 1
            continue
        if c == "(":
            depth += 1
            started = True
            i += 1
            continue
        if c == ")":
            depth -= 1
            i += 1
            if depth == 0 and started:
                j = i
                while j < n and s[j].isspace():
                    j += 1
                if j >= n or s[j] == ",":
                    calls.append(s[start:i].strip())
                    if j < n and s[j] == ",":
                        j += 1
                    start = j
                    i = j
                    started = False
                    continue
            continue
        i += 1
    tail = s[start:].strip()
    if tail:
        calls.append(tail)
    return calls


def split_call(call_str):
    if not call_str.endswith(")"):
        return None, None
    depth = 0
    open_idx = None
    for i in range(len(call_str) - 1, -1, -1):
        c = call_str[i]
        if c == ")":
            depth += 1
        elif c == "(":
            depth -= 1
            if depth == 0:
                open_idx = i
                break
    if open_idx is None:
        return None, None
    return call_str[:open_idx].strip(), call_str[open_idx + 1:-1]


def parse_calls(value):
    stripped = value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    inner = stripped[1:-1].strip()
    if not inner:
        return None
    call_strs = split_calls(inner)
    if not call_strs:
        return None

    results = []
    for call_str in call_strs:
        name, args_str = split_call(call_str)
        if name is None:
            return None
        arguments = {}
        if args_str.strip():
            for pair in split_top_level(args_str, ","):
                if "=" not in pair:
                    return None
                key, _, val_str = pair.partition("=")
                key = key.strip()
                val_str = val_str.strip()
                try:
                    val = ast.literal_eval(val_str)
                except (ValueError, SyntaxError):
                    val = val_str
                arguments[key] = val
        results.append({"name": name, "arguments": arguments})
    return results


def to_json_schema(schema):
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if out.get("type") == "dict":
        out["type"] = "object"
    if "properties" in out and isinstance(out["properties"], dict):
        out["properties"] = {k: to_json_schema(v) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = to_json_schema(out["items"])
    return out


def extract_tools(system_text):
    if not system_text:
        return system_text, []
    match = re.search(r"(\[\s*\{.*\}\s*\])", system_text, flags=re.DOTALL)
    if not match:
        return system_text, []
    try:
        raw_tools = json.loads(match.group(1))
    except json.JSONDecodeError:
        return system_text, []

    normalized = []
    for t in raw_tools:
        if not isinstance(t, dict) or "name" not in t:
            continue
        func = {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": to_json_schema(t.get("parameters", {"type": "object", "properties": {}})),
        }
        normalized.append({"type": "function", "function": func})

    cleaned = (system_text[: match.start()] + system_text[match.end():]).strip()
    cleaned = re.sub(r"invoke:\s*\n?\s*\.", "invoke the relevant tool(s).", cleaned)
    return cleaned, normalized


def convert(example):
    system_text = example.get("system", "") or ""
    clean_system, tools = extract_tools(system_text)

    messages = []
    if clean_system:
        messages.append({"role": "system", "content": clean_system})

    for turn in example["conversations"]:
        role = turn["from"]
        value = turn["value"]
        if role == "assistant":
            calls = parse_calls(value)
            if calls:
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                        for c in calls
                    ],
                })
            else:
                messages.append({"role": "assistant", "content": value})
        else:
            messages.append({"role": role, "content": value})

    return {"messages": json.dumps(messages), "tools": json.dumps(tools)}


def main():
    from datasets import load_dataset, DatasetDict

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="Team-ACE/ToolACE")
    parser.add_argument("--output_dir", default="data/toolace_processed")
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = load_dataset(args.dataset_name, split="train")
    converted = raw.map(convert, remove_columns=raw.column_names)

    split = converted.train_test_split(test_size=args.val_ratio, seed=args.seed)
    dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out_dir))
    print(f"train={len(dataset['train'])} validation={len(dataset['validation'])}")


if __name__ == "__main__":
    main()
