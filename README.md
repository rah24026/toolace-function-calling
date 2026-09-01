# Function calling fine-tune

Fine-tunes Qwen2.5-7B-Instruct for function calling on ToolACE, quantizes it, serves it with vLLM, and evaluates on BFCL.

## Setup

Training and serving use separate environments, since vLLM and the training stack pin conflicting versions of some shared dependencies.

```
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install -r requirements-train.txt
```

## Train

```
python data/prepare_data.py

python train/train.py --method lora
python train/train.py --method qlora
python train/train.py --method full

python train/merge.py --base_model Qwen/Qwen2.5-7B-Instruct --adapter_dir outputs/qwen-toolace/final --output_dir outputs/qwen-toolace/merged
```

## Serve and evaluate

```
deactivate
python3 -m venv .venv-serve
source .venv-serve/bin/activate
pip install -r requirements-serve.txt

python serve/quantize.py --method fp8 --model_dir outputs/qwen-toolace/merged --output_dir outputs/qwen-toolace/fp8
python serve/quantize.py --method awq --model_dir outputs/qwen-toolace/merged --output_dir outputs/qwen-toolace/awq

./serve/start_vllm.sh outputs/qwen-toolace/fp8 qwen25-7b-toolace
```

In another terminal, once the server is up:

```
source .venv-serve/bin/activate

git clone https://github.com/ShishirPatil/gorilla.git ../gorilla
cd ../gorilla/berkeley-function-call-leaderboard && pip install -e . && cd -

./eval/run_bfcl.sh ../gorilla/berkeley-function-call-leaderboard qwen25-7b-toolace-FC qwen25-7b-toolace

python bench/bench.py --model qwen25-7b-toolace --concurrency_levels 16 32
```

Results land in `outputs/` (training metrics), the BFCL repo's `score/` folder, and `results/benchmark_results.json`.
