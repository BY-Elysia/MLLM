import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    from torch.utils.data import Dataset, Subset
except ModuleNotFoundError:
    class Dataset:
        pass

    class Subset:
        def __init__(self, dataset: Any, indices: list[int]) -> None:
            self.dataset = dataset
            self.indices = indices

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, index: int) -> Any:
            return self.dataset[self.indices[index]]


IMAGE_PLACEHOLDER = "<image>"


def clean_user_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.replace(f"{IMAGE_PLACEHOLDER}\n", "", 1)
    text = text.replace(IMAGE_PLACEHOLDER, "").strip()
    return text


def build_clip_text(record: dict[str, Any], text_mode: str = "assistant") -> str:
    user_text = clean_user_text(record.get("user"))
    assistant_text = (record.get("assistant") or "").strip()

    if text_mode == "assistant":
        text = assistant_text or user_text
    elif text_mode == "user":
        text = user_text or assistant_text
    elif text_mode == "qa":
        parts = [part for part in [user_text, assistant_text] if part]
        text = "\n".join(parts)
    elif text_mode == "assistant_with_question":
        if assistant_text and user_text:
            text = f"Question: {user_text}\nAnswer: {assistant_text}"
        else:
            text = assistant_text or user_text
    else:
        raise ValueError(
            f"Unsupported text_mode={text_mode!r}. Expected one of "
            "'assistant', 'user', 'qa', 'assistant_with_question'."
        )

    if not text:
        raise ValueError(f"Record {record.get('id')} does not contain usable text.")
    return text


@dataclass
class CLIPSample:
    sample_id: str
    image_path: str
    text: str
    record: dict[str, Any]


class CLIPJsonlDataset(Dataset):
    def __init__(
        self,
        annotations_path: str | Path,
        dataset_root: str | Path | None = None,
        text_mode: str = "assistant",
        drop_missing_images: bool = True,
    ) -> None:
        self.annotations_path = Path(annotations_path)
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.text_mode = text_mode
        self.drop_missing_images = drop_missing_images
        self.records = self._load_records()

    def _resolve_image_path(self, image_value: str) -> Path:
        image_path = Path(image_value)
        if image_path.is_absolute():
            return image_path

        if self.dataset_root is not None:
            return self.dataset_root / image_path

        return self.annotations_path.parent / image_path

    def _load_records(self) -> list[CLIPSample]:
        samples: list[CLIPSample] = []
        with self.annotations_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                image_value = record.get("image")
                if not image_value:
                    continue

                image_path = self._resolve_image_path(image_value)
                if self.drop_missing_images and not image_path.exists():
                    continue

                try:
                    text = build_clip_text(record, text_mode=self.text_mode)
                except ValueError:
                    continue
                samples.append(
                    CLIPSample(
                        sample_id=str(record.get("id", "")),
                        image_path=str(image_path),
                        text=text,
                        record=record,
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> CLIPSample:
        return self.records[index]


class CLIPBatchCollator:
    def __init__(self, processor: Any, max_length: Optional[int] = None) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: list[CLIPSample]) -> dict[str, Any]:
        from PIL import Image

        images = []
        texts = []
        sample_ids = []
        image_paths = []
        records = []

        for sample in batch:
            with Image.open(sample.image_path) as image:
                images.append(image.convert("RGB"))
            texts.append(sample.text)
            sample_ids.append(sample.sample_id)
            image_paths.append(sample.image_path)
            records.append(sample.record)

        processor_kwargs = {
            "text": texts,
            "images": images,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        }
        if self.max_length is not None:
            processor_kwargs["max_length"] = self.max_length

        encoded = self.processor(**processor_kwargs)
        encoded["sample_ids"] = sample_ids
        encoded["texts"] = texts
        encoded["image_paths"] = image_paths
        encoded["records"] = records
        return encoded


def split_train_val_dataset(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Optional[Dataset]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0).")

    if val_ratio == 0.0 or len(dataset) < 2:
        return dataset, None

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = max(1, int(len(indices) * val_ratio))
    train_size = len(indices) - val_size
    if train_size <= 0:
        raise ValueError("val_ratio is too large for the current dataset size.")

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)
