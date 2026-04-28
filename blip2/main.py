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

from clip.data import (
    CLIPBatchCollator,
    CLIPJsonlDataset,
    CLIPSample,
    clean_user_text,
    split_train_val_dataset,
)

try:
    from .model import BLIP2Stage1Model, BLIP2Stage1Output, BLIP2Stage2Model, BLIP2Stage2Output
except ImportError:
    from model import BLIP2Stage1Model, BLIP2Stage1Output, BLIP2Stage2Model, BLIP2Stage2Output


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
    parser = argparse.ArgumentParser(
        description="Train or evaluate a BLIP-2 model with stage-1 or stage-2 style objectives."
    )
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
        "--training-stage",
        type=str,
        default=defaults.get("training_stage", "stage1"),
        choices=["stage1", "stage2"],
        help="Use BLIP-2 stage-1 style pretraining losses or stage-2 generative training.",
    )
    parser.add_argument(
        "--text-mode",
        type=str,
        default=defaults.get("text_mode", "assistant"),
        choices=["assistant", "user", "qa", "assistant_with_question"],
        help="How to build the text paired with each image for stage-1 style training.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=defaults.get("model_name", "Salesforce/blip2-itm-vit-g"),
        help="Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=path_default(defaults, "output_dir") or Path("outputs/blip2"),
    )
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 5))
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 4))
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=defaults.get("eval_batch_size", 4),
    )
    parser.add_argument("--num-workers", type=int, default=defaults.get("num_workers", 4))
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=defaults.get("learning_rate", 1e-5),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=defaults.get("weight_decay", 0.01),
    )
    parser.add_argument("--val-ratio", type=float, default=defaults.get("val_ratio", 0.1))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--max-length", type=int, default=defaults.get("max_length", 128))
    parser.add_argument(
        "--target-max-length",
        type=int,
        default=defaults.get("target_max_length", 128),
        help="Maximum target length used by stage-2 generation training.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=defaults.get("log_interval", 20),
    )
    parser.add_argument(
        "--save-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("save_every_epoch", False),
    )
    parser.add_argument(
        "--save-optimizer-state",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("save_optimizer_state", False),
        help="Whether to store optimizer and scheduler state in training_state.pt.",
    )
    parser.add_argument(
        "--freeze-vision",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("freeze_vision", True),
    )
    parser.add_argument(
        "--freeze-qformer",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("freeze_qformer", False),
    )
    parser.add_argument(
        "--freeze-text-embeddings",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("freeze_text_embeddings", False),
    )
    parser.add_argument(
        "--freeze-projection",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("freeze_projection", False),
    )
    parser.add_argument(
        "--freeze-language-model",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("freeze_language_model", True),
    )
    parser.add_argument(
        "--freeze-language-projection",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("freeze_language_projection", False),
    )
    parser.add_argument(
        "--itc-weight",
        type=float,
        default=defaults.get("itc_weight", 1.0),
        help="Weight for the stage-1 image-text contrastive loss.",
    )
    parser.add_argument(
        "--itm-weight",
        type=float,
        default=defaults.get("itm_weight", 1.0),
        help="Weight for the stage-1 image-text matching loss.",
    )
    parser.add_argument(
        "--itg-weight",
        type=float,
        default=defaults.get("itg_weight", 1.0),
        help="Weight for the stage-1 image-grounded text generation loss.",
    )
    parser.add_argument(
        "--generation-prompt-template",
        type=str,
        default=defaults.get("generation_prompt_template", "Question: {user}\nAnswer:"),
        help="Stage-2 prompt template. Supports the placeholder {user}.",
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
    args = parser.parse_args(remaining)
    args.config = config_args.config
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


class BLIP2Stage2BatchCollator:
    def __init__(
        self,
        processor: Any,
        decoder_only_language_model: bool,
        max_length: int,
        target_max_length: int,
        prompt_template: str,
    ) -> None:
        self.processor = processor
        self.decoder_only_language_model = decoder_only_language_model
        self.max_length = max_length
        self.target_max_length = target_max_length
        self.prompt_template = prompt_template

    def __call__(self, batch: list[CLIPSample]) -> dict[str, Any]:
        from PIL import Image

        images = []
        prompts = []
        targets = []
        sample_ids = []
        image_paths = []
        records = []

        for sample in batch:
            with Image.open(sample.image_path) as image:
                images.append(image.convert("RGB"))
            prompts.append(self._build_prompt(sample))
            targets.append(self._build_target(sample))
            sample_ids.append(sample.sample_id)
            image_paths.append(sample.image_path)
            records.append(sample.record)

        if not hasattr(self.processor, "tokenizer"):
            raise TypeError("Stage-2 BLIP-2 training requires a processor with a tokenizer.")

        if self.decoder_only_language_model:
            encoded = self._encode_decoder_only(images, prompts, targets)
        else:
            encoded = self._encode_encoder_decoder(images, prompts, targets)

        encoded["sample_ids"] = sample_ids
        encoded["texts"] = targets
        encoded["prompts"] = prompts
        encoded["image_paths"] = image_paths
        encoded["records"] = records
        return encoded

    def _build_prompt(self, sample: CLIPSample) -> str:
        user_text = clean_user_text(sample.record.get("user"))
        if user_text:
            return self.prompt_template.format(user=user_text)
        return "Describe the image."

    def _build_target(self, sample: CLIPSample) -> str:
        assistant_text = (sample.record.get("assistant") or "").strip()
        if assistant_text:
            return assistant_text
        return sample.text

    def _encode_encoder_decoder(
        self,
        images: list[Any],
        prompts: list[str],
        targets: list[str],
    ) -> dict[str, Any]:
        encoded = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = self.processor.tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.target_max_length,
        )["input_ids"]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        encoded["labels"] = labels
        return encoded

    def _encode_decoder_only(
        self,
        images: list[Any],
        prompts: list[str],
        targets: list[str],
    ) -> dict[str, Any]:
        full_texts = [
            f"{prompt} {target}".strip()
            for prompt, target in zip(prompts, targets)
        ]
        encoded = self.processor(
            images=images,
            text=full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_tokens = self.processor.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        prompt_lengths = prompt_tokens["attention_mask"].sum(dim=1)
        for row_index, prompt_length in enumerate(prompt_lengths.tolist()):
            labels[row_index, :prompt_length] = -100
        image_token_id = self._get_image_token_id()
        if image_token_id is not None:
            labels[encoded["input_ids"] == image_token_id] = -100
        encoded["labels"] = labels
        return encoded

    def _get_image_token_id(self) -> Optional[int]:
        tokenizer = self.processor.tokenizer
        try:
            image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        except Exception:
            return None
        if image_token_id is None or image_token_id == tokenizer.unk_token_id:
            return None
        return int(image_token_id)


def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    processor: Any,
    args: argparse.Namespace,
    *,
    decoder_only_language_model: bool = False,
) -> tuple[DataLoader, Optional[DataLoader]]:
    if args.training_stage == "stage1":
        collator: Any = CLIPBatchCollator(processor=processor, max_length=args.max_length)
    else:
        collator = BLIP2Stage2BatchCollator(
            processor=processor,
            decoder_only_language_model=decoder_only_language_model,
            max_length=args.max_length,
            target_max_length=args.target_max_length,
            prompt_template=args.generation_prompt_template,
        )

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


def compute_retrieval_accuracy(logits: torch.Tensor) -> float:
    labels = torch.arange(logits.size(0), device=logits.device)
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def compute_stage2_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    decoder_only_language_model: bool,
) -> float:
    if decoder_only_language_model:
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]

    predictions = logits.argmax(dim=-1)
    valid_mask = labels != -100
    valid_count = valid_mask.sum().item()
    if valid_count == 0:
        return 0.0
    correct = ((predictions == labels) & valid_mask).sum().item()
    return correct / valid_count


def format_metric_summary(metrics: dict[str, float]) -> str:
    ordered = []
    for key in sorted(metrics):
        ordered.append(f"{key}={metrics[key]:.4f}")
    return " ".join(ordered)


def run_stage1_epoch(
    model: BLIP2Stage1Model,
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

    total_metrics = {
        "loss": 0.0,
        "image_acc": 0.0,
        "text_acc": 0.0,
        "itc_loss": 0.0,
        "itm_loss": 0.0,
        "itg_loss": 0.0,
    }
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
            raise RuntimeError("Model did not return a loss during stage-1 BLIP-2 training.")

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_metrics["loss"] += loss.item()
        total_metrics["image_acc"] += compute_retrieval_accuracy(output.logits_per_image)
        total_metrics["text_acc"] += compute_retrieval_accuracy(output.logits_per_text)
        total_metrics["itc_loss"] += 0.0 if output.itc_loss is None else output.itc_loss.item()
        total_metrics["itm_loss"] += 0.0 if output.itm_loss is None else output.itm_loss.item()
        total_metrics["itg_loss"] += 0.0 if output.itg_loss is None else output.itg_loss.item()
        total_steps += 1

        if training and (step % log_interval == 0 or step == len(loader)):
            elapsed = time.time() - start_time
            averaged = {
                key: value / total_steps
                for key, value in total_metrics.items()
            }
            print(
                f"[Epoch {epoch}/{epochs}] "
                f"step={step}/{len(loader)} "
                f"{format_metric_summary(averaged)} "
                f"time={elapsed:.1f}s"
            )

    if total_steps == 0:
        raise RuntimeError("Dataloader is empty. Check the annotation path and dataset filters.")

    return {key: value / total_steps for key, value in total_metrics.items()}


def run_stage2_epoch(
    model: BLIP2Stage2Model,
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
    total_token_acc = 0.0
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
                labels=batch.get("labels"),
                decoder_input_ids=batch.get("decoder_input_ids"),
                decoder_attention_mask=batch.get("decoder_attention_mask"),
                return_dict=True,
            )
            loss = output.loss

        if loss is None:
            raise RuntimeError("Model did not return a loss during stage-2 BLIP-2 training.")

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_token_acc += compute_stage2_token_accuracy(
            logits=output.logits,
            labels=batch["labels"],
            decoder_only_language_model=model.use_decoder_only_language_model,
        )
        total_steps += 1

        if training and (step % log_interval == 0 or step == len(loader)):
            elapsed = time.time() - start_time
            print(
                f"[Epoch {epoch}/{epochs}] "
                f"step={step}/{len(loader)} "
                f"loss={total_loss / total_steps:.4f} "
                f"token_acc={total_token_acc / total_steps:.4f} "
                f"time={elapsed:.1f}s"
            )

    if total_steps == 0:
        raise RuntimeError("Dataloader is empty. Check the annotation path and dataset filters.")

    return {
        "loss": total_loss / total_steps,
        "token_acc": total_token_acc / total_steps,
    }


@torch.no_grad()
def evaluate_stage1(
    model: BLIP2Stage1Model,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    epochs: int,
    amp_enabled: bool,
) -> dict[str, float]:
    metrics = run_stage1_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        device=device,
        epoch=epoch,
        epochs=epochs,
        amp_enabled=amp_enabled,
        log_interval=max(1, len(loader)),
    )
    print(f"[Eval {epoch}/{epochs}] {format_metric_summary(metrics)}")
    return metrics


@torch.no_grad()
def evaluate_stage2(
    model: BLIP2Stage2Model,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    epochs: int,
    amp_enabled: bool,
) -> dict[str, float]:
    metrics = run_stage2_epoch(
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
        f"token_acc={metrics['token_acc']:.4f}"
    )
    return metrics


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    processor: Any,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    metrics: dict[str, float],
    training_stage: str,
    is_best: bool = False,
    checkpoint_name: Optional[str] = None,
    save_optimizer_state: bool = False,
) -> None:
    if checkpoint_name is None:
        checkpoint_name = "best" if is_best else f"epoch-{epoch:03d}"
    checkpoint_dir = output_dir / "checkpoints" / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(checkpoint_dir / "model")
    processor.save_pretrained(checkpoint_dir / "processor")
    training_state = {
        "epoch": epoch,
        "metrics": metrics,
        "training_stage": training_stage,
        "model_class": type(model).__name__,
    }
    if hasattr(model, "normalize"):
        training_state["normalize"] = getattr(model, "normalize")
    if save_optimizer_state:
        training_state["optimizer_state_dict"] = optimizer.state_dict()
        training_state["scheduler_state_dict"] = scheduler.state_dict()

    state_path = checkpoint_dir / "training_state.pt"
    temp_path = checkpoint_dir / "training_state.pt.tmp"
    try:
        torch.save(
            training_state,
            temp_path,
            _use_new_zipfile_serialization=False,
        )
        temp_path.replace(state_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def try_save_checkpoint(
    *,
    output_dir: Path,
    model: nn.Module,
    processor: Any,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    metrics: dict[str, float],
    training_stage: str,
    is_best: bool = False,
    checkpoint_name: Optional[str] = None,
    save_optimizer_state: bool = False,
) -> bool:
    try:
        save_checkpoint(
            output_dir=output_dir,
            model=model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=metrics,
            training_stage=training_stage,
            is_best=is_best,
            checkpoint_name=checkpoint_name,
            save_optimizer_state=save_optimizer_state,
        )
    except Exception as exc:
        target_name = checkpoint_name or ("best" if is_best else f"epoch-{epoch:03d}")
        log(f"Checkpoint save failed for {target_name}: {type(exc).__name__}: {exc}")
        return False
    return True


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


def build_model(args: argparse.Namespace, device: torch.device) -> BLIP2Stage1Model | BLIP2Stage2Model:
    if args.training_stage == "stage1":
        log(f"Loading BLIP-2 stage-1 backbone from {args.model_name}")
        model = BLIP2Stage1Model.from_pretrained(
            args.model_name,
            train_vision=not args.freeze_vision,
            train_qformer=not args.freeze_qformer,
            train_text_embeddings=not args.freeze_text_embeddings,
            train_projection=not args.freeze_projection,
            itc_weight=args.itc_weight,
            itm_weight=args.itm_weight,
            itg_weight=args.itg_weight,
        )
    else:
        log(f"Loading BLIP-2 stage-2 model from {args.model_name}")
        model = BLIP2Stage2Model.from_pretrained(
            args.model_name,
            train_vision=not args.freeze_vision,
            train_qformer=not args.freeze_qformer,
            train_language_model=not args.freeze_language_model,
            train_language_projection=not args.freeze_language_projection,
        )
    return model.to(device)


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = not args.disable_amp

    log(f"Using device: {device}")
    log(f"Loading processor from {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)

    log("Building datasets")
    train_dataset, val_dataset = build_datasets(args)

    model = build_model(args, device=device)

    log("Building dataloaders")
    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        processor,
        args,
        decoder_only_language_model=(
            isinstance(model, BLIP2Stage2Model) and model.use_decoder_only_language_model
        ),
    )

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
        f"stage={args.training_stage} device={device} "
        f"train_samples={len(train_dataset)} "
        f"val_samples={0 if val_dataset is None else len(val_dataset)} "
        f"model={args.model_name}"
    )

    for epoch in range(1, args.epochs + 1):
        log(f"Starting epoch {epoch}/{args.epochs}")
        if args.training_stage == "stage1":
            train_metrics = run_stage1_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                epochs=args.epochs,
                amp_enabled=amp_enabled,
                log_interval=args.log_interval,
            )
        else:
            train_metrics = run_stage2_epoch(
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
            if args.training_stage == "stage1":
                val_metrics = evaluate_stage1(
                    model=model,
                    loader=val_loader,
                    device=device,
                    epoch=epoch,
                    epochs=args.epochs,
                    amp_enabled=amp_enabled,
                )
            else:
                val_metrics = evaluate_stage2(
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
                try_save_checkpoint(
                    output_dir=args.output_dir,
                    model=model,
                    processor=processor,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=metrics,
                    training_stage=args.training_stage,
                    is_best=True,
                    save_optimizer_state=args.save_optimizer_state,
                )

        if args.save_every_epoch:
            try_save_checkpoint(
                output_dir=args.output_dir,
                model=model,
                processor=processor,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                training_stage=args.training_stage,
                is_best=False,
                save_optimizer_state=args.save_optimizer_state,
            )

    if val_loader is None and last_metrics is not None:
        best_metrics = last_metrics
        try_save_checkpoint(
            output_dir=args.output_dir,
            model=model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=args.epochs,
            metrics=last_metrics,
            training_stage=args.training_stage,
            checkpoint_name="final",
            save_optimizer_state=args.save_optimizer_state,
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
