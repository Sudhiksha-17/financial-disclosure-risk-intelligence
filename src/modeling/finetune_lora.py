"""
LoRA Fine-tuning for Risk Classification
==========================================
Fine-tunes Llama-3 8B using LoRA via Unsloth on the
human-annotated risk classification examples.

Single-dimension classification task: given two filing texts,
classify one risk dimension as escalating, stable, or de-escalating.

Hardware: RTX 4050 8GB VRAM
Expected training time: 20-40 minutes

Usage:
    python src/modeling/finetune_lora.py
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME    = "unsloth/Meta-Llama-3-8B-Instruct-bnb-4bit"
OUTPUT_DIR    = Path("outputs/lora_model")
TRAIN_FILE    = Path("outputs/finetune_train.jsonl")
VAL_FILE      = Path("outputs/finetune_val.jsonl")
MAX_SEQ_LEN   = 4096
LORA_RANK     = 16
BATCH_SIZE    = 1
GRAD_ACCUM    = 4
EPOCHS        = 3
LEARNING_RATE = 2e-4


# ── Data loader ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def format_example(example: dict, tokenizer) -> dict:
    """
    Format as instruction-following conversation
    using Llama-3 chat template.
    """
    messages = [
        {
            "role":    "user",
            "content": example["instruction"]
        },
        {
            "role":    "assistant",
            "content": example["output"]
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}


# ── Main ──────────────────────────────────────────────────────────────────────

def run_finetuning():
    print("=" * 60)
    print("LoRA Fine-tuning: Risk Classification")
    print(f"Model      : {MODEL_NAME}")
    print(f"Train data : {TRAIN_FILE}")
    print(f"Val data   : {VAL_FILE}")
    print(f"LoRA rank  : {LORA_RANK}")
    print(f"Epochs     : {EPOCHS}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer with Unsloth
    print("\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LEN,
        load_in_4bit   = True,
        dtype          = None,
    )

    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r                   = LORA_RANK,
        target_modules      = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha          = 16,
        lora_dropout        = 0,
        bias                = "none",
        use_gradient_checkpointing = "unsloth",
        random_state        = 42,
    )

    # Load and format data
    print("Loading training data...")
    train_raw = load_jsonl(TRAIN_FILE)
    val_raw   = load_jsonl(VAL_FILE)

    print(f"Train examples: {len(train_raw)}")
    print(f"Val examples  : {len(val_raw)}")

    train_formatted = [
        format_example(ex, tokenizer) for ex in train_raw
    ]
    val_formatted = [
        format_example(ex, tokenizer) for ex in val_raw
    ]

    train_dataset = Dataset.from_list(train_formatted)
    val_dataset   = Dataset.from_list(val_formatted)

    # Training config
    training_args = SFTConfig(
        output_dir              = str(OUTPUT_DIR),
        num_train_epochs        = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate           = LEARNING_RATE,
        fp16                    = not torch.cuda.is_bf16_supported(),
        bf16                    = torch.cuda.is_bf16_supported(),
        logging_steps           = 5,
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        warmup_ratio            = 0.1,
        lr_scheduler_type       = "cosine",
        seed                    = 42,
        report_to               = "none",
        max_seq_length          = MAX_SEQ_LEN,
        dataset_text_field      = "text",
        packing                 = False,
    )

    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        args            = training_args,
    )

    # Train
    print("\nStarting training...")
    trainer_stats = trainer.train()

    print(f"\nTraining complete.")
    print(f"Train runtime : {trainer_stats.metrics['train_runtime']:.1f}s")
    print(f"Train loss    : {trainer_stats.metrics['train_loss']:.4f}")

    # Save LoRA weights
    lora_path = OUTPUT_DIR / "lora_weights"
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    print(f"\nLoRA weights saved: {lora_path}")

    # Save training stats
    stats_path = Path("outputs/finetune_stats.json")
    stats_path.write_text(
        json.dumps({
            "model":          MODEL_NAME,
            "lora_rank":      LORA_RANK,
            "epochs":         EPOCHS,
            "train_examples": len(train_raw),
            "val_examples":   len(val_raw),
            "train_loss":     trainer_stats.metrics.get("train_loss"),
            "train_runtime":  trainer_stats.metrics.get("train_runtime"),
        }, indent=2),
        encoding="utf-8"
    )
    print(f"Stats saved: {stats_path}")
    print("\nNext: run src/modeling/evaluate_lora.py")


if __name__ == "__main__":
    run_finetuning()