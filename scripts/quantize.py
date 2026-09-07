"""Quantize the merged model to 4-bit weights (W4A16) with AWQ or GPTQ.

Uses llm-compressor (the maintained successor to AutoAWQ, from the vLLM team);
the output is compressed-tensors format, which vLLM loads natively.

Calibration uses our own rendered SFT examples rather than a generic corpus,
so the activation statistics match the schema-heavy prompts the model will
actually serve.
"""

import json
import random
from pathlib import Path

import typer
from datasets import Dataset
from llmcompressor import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_SEQ_LENGTH = 2048
NUM_CALIBRATION_SAMPLES = 256


def build_recipe(method: str) -> tuple[object, dict]:
    """Return (recipe, extra oneshot kwargs) for a quantization method."""
    if method == "awq":
        from llmcompressor.modifiers.awq import AWQModifier

        return AWQModifier(scheme="W4A16", targets=["Linear"], ignore=["lm_head"]), {}
    if method == "gptq":
        from llmcompressor.modifiers.quantization import GPTQModifier

        # offload_hessians keeps GPTQ's per-layer Hessians in CPU RAM instead
        # of adding ~270MB each to GPU pressure. Nothing else may hold the
        # GPU: a resident vLLM server pins ~90% of VRAM and OOMs this step.
        recipe = GPTQModifier(
            scheme="W4A16",
            targets=["Linear"],
            ignore=["lm_head"],
            offload_hessians=True,
        )
        return recipe, {}
    raise typer.BadParameter(f"unknown method: {method}")


def main(
    method: str = "awq",
    model_dir: str = "out/merged-bf16",
    sft_path: str = "data/sft/train.jsonl",
    output_dir: str = "",
) -> None:
    output_dir = output_dir or f"out/merged-{method}"
    # device_map puts the model on the GPU: the sequential pipeline manages
    # placement itself, but pipeline="basic" calibrates wherever the model
    # sits, and a CPU forward pass is ~50s per calibration sample.
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype="bfloat16", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    rows = [json.loads(line) for line in Path(sft_path).read_text().splitlines()]
    random.Random(13).shuffle(rows)
    rows = rows[:NUM_CALIBRATION_SAMPLES]

    def render(row: dict) -> dict:
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        return tokenizer(text, max_length=MAX_SEQ_LENGTH, truncation=True)

    dataset = Dataset.from_list([render(row) for row in rows])

    recipe, extra_kwargs = build_recipe(method)
    oneshot_kwargs = {
        "max_seq_length": MAX_SEQ_LENGTH,
        "num_calibration_samples": NUM_CALIBRATION_SAMPLES,
        **extra_kwargs,
    }
    oneshot(model=model, dataset=dataset, recipe=recipe, **oneshot_kwargs)

    model.save_pretrained(output_dir, save_compressed=True)
    tokenizer.save_pretrained(output_dir)
    print(f"{method.upper()} model saved to {output_dir}")


if __name__ == "__main__":
    typer.run(main)
