import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tokenize_example(example, tokenizer, max_length):
    messages = json.loads(example["messages"])
    tools = json.loads(example["tools"]) or None

    input_ids = []
    labels = []
    prev_ids = []

    for i, msg in enumerate(messages):
        prefix_text = tokenizer.apply_chat_template(
            messages[: i + 1], tools=tools, tokenize=False, add_generation_prompt=False
        )
        ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        if ids[: len(prev_ids)] != prev_ids:
            return None
        new_ids = ids[len(prev_ids):]
        input_ids.extend(new_ids)
        if msg["role"] == "assistant":
            labels.extend(new_ids)
        else:
            labels.extend([-100] * len(new_ids))
        prev_ids = ids

    if len(input_ids) > max_length:
        return None

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def build_dataset(raw_dataset, tokenizer, max_length):
    def _map(example):
        result = tokenize_example(example, tokenizer, max_length)
        if result is None:
            return {"input_ids": [], "labels": [], "attention_mask": []}
        return result

    tokenized = raw_dataset.map(_map, remove_columns=raw_dataset.column_names)
    return tokenized.filter(lambda ex: len(ex["input_ids"]) > 0)


@dataclass
class Collator:
    tokenizer: object

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids, labels, attention_mask = [], [], []
        for f in features:
            n_pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * n_pad)
            labels.append(f["labels"] + [-100] * n_pad)
            attention_mask.append(f["attention_mask"] + [0] * n_pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def build_model(method, model_name):
    if method == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
        )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if method in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if method == "qlora":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["full", "lora", "qlora"], default="lora")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_dir", default="data/toolace_processed")
    parser.add_argument("--output_dir", default="outputs/qwen-toolace")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.method == "full":
        args.lr = 1e-5
        args.epochs = 1
        args.batch_size = 1
        args.grad_accum = 16

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_from_disk(args.data_dir)
    train_ds = build_dataset(raw["train"], tokenizer, args.max_length)
    eval_ds = build_dataset(raw["validation"], tokenizer, args.max_length)

    model = build_model(args.method, args.model_name)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit" if args.method == "full" else "adamw_torch",
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=Collator(tokenizer=tokenizer),
    )

    result = trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    metrics = result.metrics
    metrics.update({f"final_{k}": v for k, v in trainer.evaluate().items()})
    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
