# Function calling fine-tune

Fine-tunes Qwen2.5-7B-Instruct for function calling on ToolACE, quantizes it, serves it with vLLM, and evaluates on BFCL.

Uses three separate virtual environments (training, serving, BFCL) since vLLM, AutoAWQ, llm-compressor, and BFCL's own dependencies pin conflicting versions of shared packages like torch and torchvision. Mixing them in one venv breaks things.

## Train

```
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install -r requirements-train.txt
pip install flash-attn==2.7.0.post2 --no-build-isolation

python data/prepare_data.py

python train/train.py --method lora
python train/train.py --method qlora --output_dir outputs/qwen-toolace-qlora
python train/train.py --method full --output_dir outputs/qwen-toolace-full

python train/merge.py --base_model Qwen/Qwen2.5-7B-Instruct --adapter_dir outputs/qwen-toolace/final --output_dir outputs/qwen-toolace/merged
```

LoRA is the one that gets merged and shipped, based on the comparison in results.md. QLoRA and full fine-tune are run for comparison and saved to their own output dirs.

## Quantize and serve

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-serve.txt

python serve/quantize.py --method fp8 --model_dir outputs/qwen-toolace/merged --output_dir outputs/qwen-toolace/fp8
python serve/quantize.py --method awq --model_dir outputs/qwen-toolace/merged --data_dir data/toolace_processed --output_dir outputs/qwen-toolace/awq

./serve/start_vllm.sh outputs/qwen-toolace/fp8 qwen25-7b-toolace
```

This runs in the foreground and keeps the terminal busy — run it inside tmux or screen so it survives a disconnect, and leave it running for the steps below.

## Evaluate and benchmark

In a separate terminal, with the server above still running:

```
python3 -m venv .venv-bfcl
source .venv-bfcl/bin/activate

git clone https://github.com/ShishirPatil/gorilla.git ../gorilla
cd ../gorilla/berkeley-function-call-leaderboard
pip install -e .
cd -

pip install datasets soundfile

./eval/run_bfcl.sh ../gorilla/berkeley-function-call-leaderboard qwen25-7b-toolace-FC qwen25-7b-toolace outputs/qwen-toolace/fp8

python bench/bench.py --model qwen25-7b-toolace --concurrency_levels 16 32
```

To also get numbers for the AWQ checkpoint for comparison, stop the server, restart it pointing at outputs/qwen-toolace/awq with a different served name, then run the same two commands above against that name. Switch the server back to the fp8 checkpoint afterward since that's the one actually being deployed.

Results land in outputs/ (training metrics), the BFCL repo's score/ folder, and results/benchmark_results.json. See results.md for the actual numbers from these runs.
