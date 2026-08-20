import gc
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

from .data import build_training_texts, load_gsm8k
from .quantization import is_quantized, quantization_config


class CausalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.encodings.items()}
        item["labels"] = list(item["input_ids"])
        return item


def build_dataset(tokenizer, max_length: int, limit=None):
    dataset = load_gsm8k("train", limit)
    texts = build_training_texts(dataset, tokenizer.eos_token)
    encodings = tokenizer(texts, truncation=True, max_length=max_length,
                          padding=False)
    return CausalDataset({"input_ids": encodings["input_ids"],
                          "attention_mask": encodings["attention_mask"]})


def train_lora(config, model_path: str, adapter_dir: Path) -> Path:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=quantization_config(config.precision))
    model.config.use_cache = False
    if is_quantized(config.precision):
        model = prepare_model_for_kbit_training(model,
                                                use_gradient_checkpointing=False)

    model = get_peft_model(model, LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
    ))
    model.print_trainable_parameters()

    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(adapter_dir / "checkpoints"),
            per_device_train_batch_size=config.train_batch_size,
            gradient_accumulation_steps=config.train_grad_accum,
            num_train_epochs=config.train_epochs,
            learning_rate=config.train_learning_rate,
            warmup_steps=config.train_warmup_steps,
            lr_scheduler_type="linear",
            weight_decay=0.01,
            optim="adamw_torch",
            bf16=True,
            logging_steps=25,
            save_strategy="no",
            report_to="none",
            seed=config.seed,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        ),
        train_dataset=build_dataset(tokenizer, config.train_max_length,
                                   config.train_limit),
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                             padding=True),
    )

    trainer.train()
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return adapter_dir
