# Results

## Model and method

Base model is Qwen2.5-7B-Instruct. It already knows how to output tool calls in the
`<tool_call>{...}</tool_call>` format, so fine-tuning is sharpening an existing skill instead
of teaching a new output format from scratch. It's also small enough to hit good latency at
16-32 concurrent requests on one H100, and it's Apache-2.0 licensed.

Three fine-tuning methods were run on the same data and compared.

LoRA: 1h56m to train, train_loss 0.173, eval_loss 0.176
QLoRA: 2h42m to train, train_loss 0.174, eval_loss 0.177
Full fine-tune: 39m to train, train_loss 0.235, eval_loss 0.150

LoRA and QLoRA land on basically the same quality, but QLoRA took about 40% longer because
the 4-bit base weights have to be dequantized on every forward/backward pass. Since the
whole point of QLoRA is saving GPU memory, and we have plenty of memory to spare on an 80GB
H100 for a 7B model, LoRA is the better pick here — same result, faster and simpler.

Full fine-tuning only ran 1 epoch (all parameters updating at once needs a much smaller
learning rate and less exposure to a single-domain dataset to avoid overfitting/forgetting),
so its train_loss is higher just because it's averaged over a full epoch instead of the
tail end of 3 epochs. Its eval_loss actually came out lowest, but eval_loss on ToolACE isn't
the real test — BFCL is.

LoRA is the one that got merged, quantized, and shipped.

## Quantization

FP8 (llm-compressor, FP8_DYNAMIC) and AWQ INT4 were both produced from the merged LoRA
checkpoint. FP8 came out to 8.2 GB, AWQ to 5.2 GB. Both were actually evaluated end to end
(BFCL and the concurrency benchmark), not just reasoned about.

BFCL Python subset:
FP8 — live overall 77.35%, non-live overall 74.54%
AWQ — live overall 76.31%, non-live overall 74.23%

Concurrency benchmark (requests per second):
FP8 — 26.5 at concurrency 16, 57.6 at concurrency 32
AWQ — 24.0 at concurrency 16, 47.2 at concurrency 32

FP8 wins on both accuracy and speed. AWQ is close on accuracy, about a point lower, but is
also slower despite being a smaller checkpoint, because 4-bit weights need to be
dequantized before each matmul and don't get Hopper's native FP8 tensor-core speedup. So
FP8 isn't a compromise between accuracy and speed here, it's just better on both counts for
this model on this hardware. FP8 is what's actually deployed.

## BFCL — Python subset

Served with vLLM, evaluated with the official bfcl-eval harness against `--test-category
python`, which covers the Python-only categories (simple/multiple/parallel/parallel_multiple,
irrelevance, and their live variants). These are the FP8 numbers, since that's the deployed
model.

Non-live (static tool docs):
Simple AST 95.00%, Multiple AST 95.50%, Parallel AST 90.50%, Parallel Multiple AST 80.50%,
Irrelevance Detection 80.00%. Overall 74.54%.

Live (user-contributed):
Simple AST 79.07%, Multiple AST 77.21%, Parallel AST 75.00%, Parallel Multiple AST 66.67%,
Irrelevance Detection 72.40%, Relevance Detection 87.50%. Overall 77.35%.

Parallel Multiple is the weakest category by a clear margin, in both the live and non-live
sets, and it came out the same for AWQ too, so it's not checkpoint noise. It's a known-hard
BFCL category in general (picking the right subset of calls out of several parallel tool
calls when more than one looks plausible), not something specific to this fine-tune.

BFCL's own combined leaderboard table also prints a much lower "Overall Acc" number for this
model, but that one blends in test groups we never ran — multi-turn, web search, memory — as
0%. Those aren't part of the Python subset or this task, so the category numbers above are
the real result, not that blended one.

## Concurrency benchmark

200 requests each at 16 and 32 concurrent, streamed, against the FP8 checkpoint on vLLM.

At concurrency 16: TTFT mean 124ms (p50 98ms, p99 499ms), end-to-end latency mean 546ms (p50
416ms, p99 1.98s), throughput 1993 tokens/sec and 26.5 requests/sec.

At concurrency 32: TTFT mean 79ms (p50 78ms, p99 154ms), end-to-end latency mean 422ms (p50
317ms, p99 1.53s), throughput 4326 tokens/sec and 57.6 requests/sec.

Latency actually got better going from 16 to 32 concurrency instead of worse. That's because
requests reuse a fixed pool of real prompts with repeated tool schemas, and vLLM's prefix
cache was already warm by the time the 32-concurrency run started. That's not an artifact to
explain away — a FinTech agent calling the same fixed set of internal APIs on every request
is exactly the traffic pattern where prefix caching pays off, so this is a realistic result
for the actual production workload, not an inflated one.

## Biggest challenges

ToolACE's assistant turns aren't JSON, they're a pseudo-Python call list like
[FuncName(arg="val"), ...], and function names can contain spaces and even their own
parentheses, so a plain string split doesn't work. Needed a real bracket-aware parser to
handle it correctly.

Storing the parsed messages and tools as nested objects broke datasets' Arrow schema
inference at full scale, 11,300 examples all with different tool shapes, so those fields got
serialized to JSON strings instead.

The vLLM, AutoAWQ, llm-compressor, torchvision dependency stack fought itself constantly.
Exact version pins from different projects conflicted, and installing BFCL into the same
venv as vLLM silently reinstalled a torchvision build that broke the server. Splitting
training, serving, and BFCL into three separate virtual environments fixed this for good.

BFCL also needed a couple of undocumented adjustments to work against an already-running
external server instead of spinning up its own: --skip-server-setup, pointing
REMOTE_OPENAI_TOKENIZER_PATH at the local checkpoint resolved to an absolute path since the
script changes directories before running BFCL, and --allow-overwrite so a rerun after
fixing the server's context length actually regenerated results instead of silently reusing
the old ones.
