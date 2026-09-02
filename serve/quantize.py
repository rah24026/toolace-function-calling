import argparse


def quantize_fp8(model_dir, output_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    oneshot(model=model, recipe=recipe)

    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)


def quantize_awq(model_dir, data_dir, output_dir, num_calib_samples):
    import json

    from awq import AutoAWQForCausalLM
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ds = load_from_disk(data_dir)["validation"]
    n = min(num_calib_samples, len(ds))
    calib_texts = [
        tokenizer.apply_chat_template(
            json.loads(ex["messages"]), tools=json.loads(ex["tools"]) or None, tokenize=False, add_generation_prompt=False
        )
        for ex in ds.select(range(n))
    ]

    model = AutoAWQForCausalLM.from_pretrained(model_dir, safetensors=True)
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)

    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["fp8", "awq"], required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_dir", default="data/toolace_processed")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_calib_samples", type=int, default=256)
    args = parser.parse_args()

    if args.method == "fp8":
        quantize_fp8(args.model_dir, args.output_dir)
    else:
        quantize_awq(args.model_dir, args.data_dir, args.output_dir, args.num_calib_samples)


if __name__ == "__main__":
    main()
