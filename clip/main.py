import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

try:
    from .data import CLIPBatchCollator, CLIPJsonlDataset, split_train_val_dataset
    from .model import CLIPContrastiveModel
except ImportError:
    from data import CLIPBatchCollator, CLIPJsonlDataset, split_train_val_dataset
    from model import CLIPContrastiveModel


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a JSON object: {config_path}")
    return data


def path_default(defaults: dict[str, Any], key: str) -> Optional[Path]:
    value = defaults.get(key)
    if value in (None, ""):
        return None
    return Path(value)


def build_parser(defaults: Optional[dict[str, Any]] = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="Train or evaluate a CLIP contrastive model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config file. CLI arguments override config values.",
    )
    parser.add_argument(
        "--train-annotations",
        type=Path,
        default=path_default(defaults, "train_annotations"),
        help="Path to the training JSONL annotations.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=path_default(defaults, "dataset_root"),
        help="Optional root directory used to resolve relative image paths.",
    )
    parser.add_argument(
        "--val-annotations",
        type=Path,
        default=path_default(defaults, "val_annotations"),
        help="Optional validation JSONL annotations. If omitted, the train set is split.",
    )
    parser.add_argument(
        "--text-mode",
        type=str,
        default=defaults.get("text_mode", "assistant"),
        choices=["assistant", "user", "qa", "assistant_with_question"],
        help="How to build the text paired with each image.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=defaults.get("model_name", "openai/clip-vit-base-patch32"),
        help="Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=path_default(defaults, "output_dir") or Path("outputs/clip"),
    )
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 5))
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 32))
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=defaults.get("eval_batch_size", 32),
    )
    parser.add_argument("--num-workers", type=int, default=defaults.get("num_workers", 4))
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=defaults.get("learning_rate", 5e-5),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=defaults.get("weight_decay", 0.01),
    )
    parser.add_argument("--val-ratio", type=float, default=defaults.get("val_ratio", 0.1))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--max-length", type=int, default=defaults.get("max_length", 77))
    parser.add_argument("--log-interval", type=int, default=defaults.get("log_interval", 20))
    parser.add_argument(
        "--save-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("save_every_epoch", False),
    )
    parser.add_argument(
        "--freeze-vision",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("freeze_vision", False),
    )
    parser.add_argument(
        "--freeze-text",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("freeze_text", False),
    )
    parser.add_argument(
        "--freeze-projection",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("freeze_projection", False),
    )
    parser.add_argument(
        "--freeze-logit-scale",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("freeze_logit_scale", False),
    )
    parser.add_argument(
        "--disable-amp",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("disable_amp", False),
    )
    parser.add_argument("--device", type=str, default=defaults.get("device"))
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, remaining = config_parser.parse_known_args()

    defaults = {}
    if config_args.config is not None:
        defaults = load_config(config_args.config)

    parser = build_parser(defaults=defaults)
    args = parser.parse_args()
    if args.train_annotations is None:
        parser.error("--train-annotations is required unless provided in --config")
    return args


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(args: argparse.Namespace) -> tuple[Dataset, Optional[Dataset]]:
    train_dataset = CLIPJsonlDataset(
        annotations_path=args.train_annotations,
        dataset_root=args.dataset_root,
        text_mode=args.text_mode,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty after filtering missing images.")

    if args.val_annotations is not None:
        val_dataset = CLIPJsonlDataset(
            annotations_path=args.val_annotations,
            dataset_root=args.dataset_root,
            text_mode=args.text_mode,
        )
        if len(val_dataset) == 0:
            raise RuntimeError("Validation dataset is empty after filtering missing images.")
        return train_dataset, val_dataset

    return split_train_val_dataset(
        dataset=train_dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    processor: Any,
    args: argparse.Namespace,
) -> tuple[DataLoader, Optional[DataLoader]]:
    collator = CLIPBatchCollator(processor=processor, max_length=args.max_length)
    common_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collator,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **common_kwargs,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            **common_kwargs,
        )
    return train_loader, val_loader


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def compute_accuracy(logits: torch.Tensor) -> float:
    labels = torch.arange(logits.size(0), device=logits.device)
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[AdamW],
    device: torch.device,
    epoch: int,
    epochs: int,
    amp_enabled: bool,
    log_interval: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_image_acc = 0.0
    total_text_acc = 0.0
    total_steps = 0
    start_time = time.time()

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        context = (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if amp_enabled and device.type == "cuda"
            else nullcontext()
        )

        with context:
            output = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                return_loss=True,
            )
            loss = output.loss

        if loss is None:
            raise RuntimeError("Model did not return a loss during contrastive training.")

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        image_acc = compute_accuracy(output.logits_per_image)
        text_acc = compute_accuracy(output.logits_per_text)

        total_loss += loss.item()
        total_image_acc += image_acc
        total_text_acc += text_acc
        total_steps += 1

        if training and (step % log_interval == 0 or step == len(loader)):
            elapsed = time.time() - start_time
            print(
                f"[Epoch {epoch}/{epochs}] "
                f"step={step}/{len(loader)} "
                f"loss={total_loss / total_steps:.4f} "
                f"img_acc={total_image_acc / total_steps:.4f} "
                f"text_acc={total_text_acc / total_steps:.4f} "
                f"time={elapsed:.1f}s"
            )

    if total_steps == 0:
        raise RuntimeError("Dataloader is empty. Check the annotation path and dataset filters.")

    return {
        "loss": total_loss / total_steps,
        "image_acc": total_image_acc / total_steps,
        "text_acc": total_text_acc / total_steps,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    epochs: int,
    amp_enabled: bool,
) -> dict[str, float]:
    metrics = run_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        device=device,
        epoch=epoch,
        epochs=epochs,
        amp_enabled=amp_enabled,
        log_interval=max(1, len(loader)),
    )
    print(
        f"[Eval {epoch}/{epochs}] "
        f"loss={metrics['loss']:.4f} "
        f"img_acc={metrics['image_acc']:.4f} "
        f"text_acc={metrics['text_acc']:.4f}"
    )
    return metrics


def save_checkpoint(
    output_dir: Path,
    model: CLIPContrastiveModel,
    processor: Any,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    metrics: dict[str, float],
    is_best: bool = False,
    checkpoint_name: Optional[str] = None,
) -> None:
    if checkpoint_name is None:
        checkpoint_name = "best" if is_best else f"epoch-{epoch:03d}"
    checkpoint_dir = output_dir / "checkpoints" / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.clip.save_pretrained(checkpoint_dir / "clip")
    processor.save_pretrained(checkpoint_dir / "processor")
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "normalize": model.normalize,
            "max_logit_scale": model.max_logit_scale,
        },
        checkpoint_dir / "training_state.pt",
    )


def write_run_summary(
    output_dir: Path,
    args: argparse.Namespace,
    train_size: int,
    val_size: int,
    best_metrics: Optional[dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "train_size": train_size,
        "val_size": val_size,
        "best_metrics": best_metrics,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = not args.disable_amp

    log(f"Using device: {device}")
    log(f"Loading processor from {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)

    log("Building datasets")
    train_dataset, val_dataset = build_datasets(args)

    log("Building dataloaders")
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, processor, args)

    log(f"Loading CLIP model from {args.model_name}")
    model = CLIPContrastiveModel.from_pretrained(
        args.model_name,
        train_vision=not args.freeze_vision,
        train_text=not args.freeze_text,
        train_projection=not args.freeze_projection,
        train_logit_scale=not args.freeze_logit_scale,
    ).to(device)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found. Check the freeze flags.")

    optimizer = AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.learning_rate / 100.0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_metrics = None
    best_val_loss = math.inf
    last_metrics = None

    log(
        f"device={device} train_samples={len(train_dataset)} "
        f"val_samples={0 if val_dataset is None else len(val_dataset)} "
        f"model={args.model_name}"
    )

    for epoch in range(1, args.epochs + 1):
        log(f"Starting epoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
            amp_enabled=amp_enabled,
            log_interval=args.log_interval,
        )
        scheduler.step()

        metrics = {"train": train_metrics}
        last_metrics = metrics
        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                epoch=epoch,
                epochs=args.epochs,
                amp_enabled=amp_enabled,
            )
            metrics["val"] = val_metrics
            last_metrics = metrics

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_metrics = metrics
                save_checkpoint(
                    output_dir=args.output_dir,
                    model=model,
                    processor=processor,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=metrics,
                    is_best=True,
                )

        if args.save_every_epoch:
            save_checkpoint(
                output_dir=args.output_dir,
                model=model,
                processor=processor,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                is_best=False,
            )

    if val_loader is None and last_metrics is not None:
        best_metrics = last_metrics
        save_checkpoint(
            output_dir=args.output_dir,
            model=model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=args.epochs,
            metrics=last_metrics,
            checkpoint_name="final",
                )

    write_run_summary(
        output_dir=args.output_dir,
        args=args,
        train_size=len(train_dataset),
        val_size=0 if val_dataset is None else len(val_dataset),
        best_metrics=best_metrics,
    )
    log("Training finished")


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
