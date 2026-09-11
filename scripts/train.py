"""QLoRA fine-tune of Llama-3.2 3B Instruct on Spider text-to-SQL.

Runs on the GPU box (single L4):

    uv run --group train scripts/train.py --output-dir out/qlora-r16

Expects data/sft/{train,val}.jsonl produced by build_train_data.py.
"""

from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[1]


def main(
    model_id: str = "unsloth/Llama-3.2-3B-Instruct",
    output_dir: str = "out/qlora-r16",
    epochs: float = 2.0,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_seq_length: int = 4096,
    seed: int = 13,
):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(ROOT / "data/sft/train.jsonl"),
            "val": str(ROOT / "data/sft/val.jsonl"),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # TRL's conversational prompt-completion format: loss lands on the
    # completion (the SQL) only, not on re-predicting the schema text.
    def to_prompt_completion(row):
        return {"prompt": row["messages"][:-1], "completion": [row["messages"][-1]]}

    dataset = dataset.map(to_prompt_completion, remove_columns=["messages", "db_id"])

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=25,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        max_length=max_seq_length,
        completion_only_loss=True,
        packing=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        seed=seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    # Spot VMs get preempted: pick up from the newest checkpoint when one exists.
    from transformers.trainer_utils import get_last_checkpoint

    last_checkpoint = get_last_checkpoint(output_dir) if Path(output_dir).is_dir() else None
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"adapter saved to {output_dir}")


if __name__ == "__main__":
    typer.run(main)
