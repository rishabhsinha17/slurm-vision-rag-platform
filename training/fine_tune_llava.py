import os, argparse, json, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def load_data(path):
    ds = load_dataset("json", data_files=path, split="train")
    return ds

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="liuhaotian/llava-v1.5-7b")
    p.add_argument("--train_jsonl", default="data/train.jsonl")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")

    model.gradient_checkpointing_enable()
    lora = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora)

    ds = load_data(args.train_jsonl)
    def tokenize(ex):
        text = ex["text"]
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=2048)
        enc["labels"] = enc["input_ids"].copy()
        return enc
    ds = ds.map(tokenize, remove_columns=ds.column_names)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        bf16=True,
        logging_steps=10,
        report_to=[]
    )
    trainer = Trainer(model=model, args=train_args, train_dataset=ds)
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()