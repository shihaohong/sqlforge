"""Keep saved model artifacts loadable by the serving stack.

The training box runs transformers 5.x (for trl/peft), while vLLM pins
transformers 4.x. A tokenizer saved by the newer library records
`tokenizer_class: "TokenizersBackend"`, a name 4.x does not know, and vLLM
fails at startup with "Failed to load the tokenizer". The tokenizer data
itself (tokenizer.json) is fine, so naming the portable fast-tokenizer class
is enough to make one artifact loadable by both.
"""

import json
from pathlib import Path

# transformers 5.x writes this placeholder for tokenizers-backed tokenizers.
UNPORTABLE_CLASSES = {"TokenizersBackend"}
PORTABLE_CLASS = "PreTrainedTokenizerFast"


def normalize_tokenizer_config(model_dir: str | Path) -> bool:
    """Rewrite a tokenizer_class that older transformers cannot resolve.

    Returns True when the file was changed.
    """
    path = Path(model_dir) / "tokenizer_config.json"
    if not path.exists():
        return False
    config = json.loads(path.read_text())
    if config.get("tokenizer_class") not in UNPORTABLE_CLASSES:
        return False
    config["tokenizer_class"] = PORTABLE_CLASS
    path.write_text(json.dumps(config, indent=2, sort_keys=True))
    return True
