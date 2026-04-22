import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from clip.main import run_training


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "clip" / "config.json"
PATH_KEYS = {
    "train_annotations",
    "dataset_root",
    "val_annotations",
    "output_dir",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repository entrypoint for model experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the experiment config file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a JSON object: {config_path}")
    return data


def resolve_path(repo_root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def resolve_model_name(repo_root: Path, value: Any) -> str:
    if value in (None, ""):
        raise ValueError("model_name is required.")
    raw = str(value)
    candidate = (repo_root / raw).resolve()
    if candidate.exists():
        return str(candidate)
    return raw


def build_training_namespace(config: dict[str, Any], repo_root: Path) -> Namespace:
    if "train_annotations" not in config:
        raise ValueError("train_annotations is required in the config file.")

    normalized: dict[str, Any] = {}
    for key, value in config.items():
        if key in PATH_KEYS:
            normalized[key] = resolve_path(repo_root, value)
        elif key == "model_name":
            normalized[key] = resolve_model_name(repo_root, value)
        else:
            normalized[key] = value

    return Namespace(
        config=None,
        train_annotations=normalized["train_annotations"],
        dataset_root=normalized.get("dataset_root"),
        val_annotations=normalized.get("val_annotations"),
        text_mode=str(normalized.get("text_mode", "assistant")),
        model_name=normalized.get("model_name", "openai/clip-vit-base-patch32"),
        output_dir=normalized.get("output_dir") or (repo_root / "outputs/clip"),
        epochs=int(normalized.get("epochs", 5)),
        batch_size=int(normalized.get("batch_size", 32)),
        eval_batch_size=int(normalized.get("eval_batch_size", 32)),
        num_workers=int(normalized.get("num_workers", 4)),
        learning_rate=float(normalized.get("learning_rate", 5e-5)),
        weight_decay=float(normalized.get("weight_decay", 0.01)),
        val_ratio=float(normalized.get("val_ratio", 0.1)),
        seed=int(normalized.get("seed", 42)),
        max_length=int(normalized.get("max_length", 77)),
        log_interval=int(normalized.get("log_interval", 20)),
        save_every_epoch=bool(normalized.get("save_every_epoch", False)),
        freeze_vision=bool(normalized.get("freeze_vision", False)),
        freeze_text=bool(normalized.get("freeze_text", False)),
        freeze_projection=bool(normalized.get("freeze_projection", False)),
        freeze_logit_scale=bool(normalized.get("freeze_logit_scale", False)),
        disable_amp=bool(normalized.get("disable_amp", False)),
        device=normalized.get("device"),
    )


def main() -> None:
    cli_args = parse_args()
    config_path = cli_args.config.resolve()
    config = load_config(config_path)
    training_args = build_training_namespace(config, repo_root=REPO_ROOT)
    run_training(training_args)


if __name__ == "__main__":
    main()
