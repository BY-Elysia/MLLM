import argparse
import importlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "clip" / "config.json"
PATH_KEYS = {
    "train_annotations",
    "dataset_root",
    "val_annotations",
    "output_dir",
}
RUNNER_MODULES = {
    "clip": "clip.main",
    "blip2": "blip2.main",
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


def infer_model_family(config: dict[str, Any], config_path: Path) -> str:
    explicit = str(config.get("model_family", "")).strip().lower()
    if explicit:
        if explicit not in RUNNER_MODULES:
            supported = ", ".join(sorted(RUNNER_MODULES))
            raise ValueError(f"Unsupported model_family={explicit!r}. Expected one of: {supported}.")
        return explicit

    parent_name = config_path.parent.name.lower()
    if parent_name in RUNNER_MODULES:
        return parent_name

    if config_path.resolve() == DEFAULT_CONFIG_PATH.resolve():
        return "clip"

    raise ValueError(
        "Cannot infer which trainer to use from the config path. "
        "Add `model_family` to the config file."
    )


def normalize_config_paths(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
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

    return normalized


def build_training_namespace(
    config: dict[str, Any],
    repo_root: Path,
    model_family: str,
) -> Namespace:
    normalized = normalize_config_paths(config, repo_root=repo_root)

    common_kwargs = {
        "config": None,
        "model_family": model_family,
        "train_annotations": normalized["train_annotations"],
        "dataset_root": normalized.get("dataset_root"),
        "val_annotations": normalized.get("val_annotations"),
        "text_mode": str(normalized.get("text_mode", "assistant")),
        "model_name": normalized.get(
            "model_name",
            "Salesforce/blip2-itm-vit-g"
            if model_family == "blip2"
            else "openai/clip-vit-base-patch32",
        ),
        "output_dir": normalized.get("output_dir") or (repo_root / f"outputs/{model_family}"),
        "epochs": int(normalized.get("epochs", 5)),
        "batch_size": int(normalized.get("batch_size", 4 if model_family == "blip2" else 32)),
        "eval_batch_size": int(
            normalized.get("eval_batch_size", 4 if model_family == "blip2" else 32)
        ),
        "num_workers": int(normalized.get("num_workers", 4)),
        "learning_rate": float(
            normalized.get("learning_rate", 1e-5 if model_family == "blip2" else 5e-5)
        ),
        "weight_decay": float(normalized.get("weight_decay", 0.01)),
        "val_ratio": float(normalized.get("val_ratio", 0.1)),
        "seed": int(normalized.get("seed", 42)),
        "max_length": int(
            normalized.get("max_length", 128 if model_family == "blip2" else 77)
        ),
        "log_interval": int(normalized.get("log_interval", 20)),
        "save_every_epoch": bool(normalized.get("save_every_epoch", False)),
        "save_optimizer_state": bool(normalized.get("save_optimizer_state", False)),
        "freeze_vision": bool(normalized.get("freeze_vision", model_family == "blip2")),
        "freeze_projection": bool(normalized.get("freeze_projection", False)),
        "disable_amp": bool(normalized.get("disable_amp", False)),
        "device": normalized.get("device"),
    }

    if model_family == "clip":
        common_kwargs["freeze_text"] = bool(normalized.get("freeze_text", False))
        common_kwargs["freeze_logit_scale"] = bool(normalized.get("freeze_logit_scale", False))
    elif model_family == "blip2":
        common_kwargs["training_stage"] = str(normalized.get("training_stage", "stage1"))
        common_kwargs["freeze_qformer"] = bool(normalized.get("freeze_qformer", False))
        common_kwargs["freeze_text_embeddings"] = bool(
            normalized.get("freeze_text_embeddings", normalized.get("freeze_text", False))
        )
        common_kwargs["freeze_language_model"] = bool(
            normalized.get("freeze_language_model", True)
        )
        common_kwargs["freeze_language_projection"] = bool(
            normalized.get("freeze_language_projection", False)
        )
        common_kwargs["itc_weight"] = float(normalized.get("itc_weight", 1.0))
        common_kwargs["itm_weight"] = float(normalized.get("itm_weight", 1.0))
        common_kwargs["itg_weight"] = float(normalized.get("itg_weight", 1.0))
        common_kwargs["target_max_length"] = int(normalized.get("target_max_length", 128))
        common_kwargs["generation_prompt_template"] = str(
            normalized.get("generation_prompt_template", "Question: {user}\nAnswer:")
        )

    return Namespace(**common_kwargs)


def load_runner(model_family: str) -> Any:
    module_name = RUNNER_MODULES[model_family]
    module = importlib.import_module(module_name)
    return module.run_training


def main() -> None:
    cli_args = parse_args()
    config_path = cli_args.config.resolve()
    config = load_config(config_path)
    model_family = infer_model_family(config=config, config_path=config_path)
    training_args = build_training_namespace(
        config,
        repo_root=REPO_ROOT,
        model_family=model_family,
    )
    run_training = load_runner(model_family)
    run_training(training_args)


if __name__ == "__main__":
    main()
