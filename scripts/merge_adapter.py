"""Merge the QLoRA adapter into the base model as bf16 weights.

The adapter was trained against the 4-bit NF4 base, so merging into bf16
introduces a small train/serve mismatch; the eval gate afterwards measures
whether it matters.
"""

from pathlib import Path

import torch
import typer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from text2sql.artifacts import normalize_tokenizer_config

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"


def main(
    adapter_dir: str = "out/qlora-r16",
    output_dir: str = "out/merged-bf16",
) -> None:
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)

    # The adapter dir carries the tokenizer + chat template used in training.
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)
    normalize_tokenizer_config(output_dir)
    print(f"merged model saved to {output_dir}")


if __name__ == "__main__":
    typer.run(main)
