import json

from text2sql.artifacts import normalize_tokenizer_config


def write_config(tmp_path, config: dict):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    return tmp_path


def test_rewrites_the_unportable_class(tmp_path):
    write_config(tmp_path, {"tokenizer_class": "TokenizersBackend", "backend": "tokenizers"})
    assert normalize_tokenizer_config(tmp_path)
    config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert config["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert config["backend"] == "tokenizers"  # everything else is preserved


def test_leaves_a_known_class_alone(tmp_path):
    write_config(tmp_path, {"tokenizer_class": "PreTrainedTokenizerFast"})
    assert not normalize_tokenizer_config(tmp_path)


def test_tolerates_a_missing_config(tmp_path):
    assert not normalize_tokenizer_config(tmp_path)
