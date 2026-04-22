import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from clip.main import run_training
from clip.make_subset import make_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repository entrypoint for model experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to the single experiment config file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a JSON object: {config_path}")
    return data


def resolve_path(base_dir: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resolve_model_name(base_dir: Path, value: Any) -> str:
    if value in (None, ""):
        raise ValueError("train.model_name is required.")
    raw = str(value)
    candidate = (base_dir / raw).resolve()
    if candidate.exists():
        return str(candidate)
    return raw


def build_training_namespace(config: dict[str, Any], repo_root: Path) -> Namespace:
    if str(config.get("task", "clip")).strip().lower() != "clip":
        raise ValueError("Only task='clip' is supported right now.")

    subset_cfg = config.get("subset") or {}
    train_cfg = config.get("train") or {}

    if not isinstance(subset_cfg, dict) or not isinstance(train_cfg, dict):
        raise TypeError("Config sections 'subset' and 'train' must be JSON objects.")

    train_annotations = resolve_path(repo_root, train_cfg.get("train_annotations"))

    if bool(subset_cfg.get("enabled", False)):
        source_annotations = resolve_path(repo_root, subset_cfg.get("source_annotations"))
        output_annotations = resolve_path(repo_root, subset_cfg.get("output_annotations"))
        if source_annotations is None or output_annotations is None:
            raise ValueError(
                "subset.source_annotations and subset.output_annotations are required when subset.enabled=true."
            )
        subset_result = make_subset(
            input_path=source_annotations,
            output_path=output_annotations,
            limit=int(subset_cfg.get("limit", 64)),
            seed=int(subset_cfg.get("seed", 42)),
        )
        print(json.dumps(subset_result, ensure_ascii=False, indent=2), flush=True)
        train_annotations = output_annotations

    if train_annotations is None:
        raise ValueError("train.train_annotations is required.")

    args = Namespace(
        config=None,
        train_annotations=train_annotations,
        dataset_root=resolve_path(repo_root, train_cfg.get("dataset_root")),
        val_annotations=resolve_path(repo_root, train_cfg.get("val_annotations")),
        text_mode=str(train_cfg.get("text_mode", "assistant")),
        model_name=resolve_model_name(repo_root, train_cfg.get("model_name")),
        output_dir=resolve_path(repo_root, train_cfg.get("output_dir")) or (repo_root / "outputs/clip"),
        epochs=int(train_cfg.get("epochs", 5)),
        batch_size=int(train_cfg.get("batch_size", 32)),
        eval_batch_size=int(train_cfg.get("eval_batch_size", 32)),
        num_workers=int(train_cfg.get("num_workers", 4)),
        learning_rate=float(train_cfg.get("learning_rate", 5e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        val_ratio=float(train_cfg.get("val_ratio", 0.1)),
        seed=int(train_cfg.get("seed", 42)),
        max_length=int(train_cfg.get("max_length", 77)),
        log_interval=int(train_cfg.get("log_interval", 20)),
        save_every_epoch=bool(train_cfg.get("save_every_epoch", False)),
        freeze_vision=bool(train_cfg.get("freeze_vision", False)),
        freeze_text=bool(train_cfg.get("freeze_text", False)),
        freeze_projection=bool(train_cfg.get("freeze_projection", False)),
        freeze_logit_scale=bool(train_cfg.get("freeze_logit_scale", False)),
        disable_amp=bool(train_cfg.get("disable_amp", False)),
        device=train_cfg.get("device"),
    )
    return args


def main() -> None:
    cli_args = parse_args()
    config_path = cli_args.config.resolve()
    repo_root = config_path.parent.resolve()
    config = load_config(config_path)
    training_args = build_training_namespace(config, repo_root=repo_root)
    run_training(training_args)


if __name__ == "__main__":
    main()
