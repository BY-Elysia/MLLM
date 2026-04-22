#!/usr/bin/env python3
import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_rows(manifest_path: Path) -> list[dict]:
    rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def copy_one(src: Path, dst: Path) -> tuple[str, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst), dst.stat().st_size


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_readme(
    dataset_dir: Path,
    rows: list[dict],
    image_bytes: int,
    source_manifest: Path,
    source_root: Path,
) -> None:
    readme = f"""# UReader Existing Images Subset

This directory contains the currently usable subset from `ureader_kg_processed.json`.

- Samples: {len(rows)}
- Image files: {len(rows)}
- Image size: {image_bytes} bytes
- Image size (GiB): {image_bytes / 1024 / 1024 / 1024:.3f}
- Source manifest: {source_manifest}
- Source image root: {source_root}

## Structure

- `images/`: copied local images
- `annotations/train.jsonl`: cleaned annotations using local relative image paths

## Record Format

Each line in `annotations/train.jsonl` is a JSON object like:

```json
{{
  "id": "00326327001359",
  "data_source": "ureader_kg",
  "image": "images/ureader-instruction-1.0/ChartQA/train/png/00326327001359.png",
  "user": "<image>\\nList a handful of essential elements in this visual.",
  "assistant": "There are two categories in the chart. The rate in Guyana is not twice that of Papua New Guinea.",
  "messages": [
    {{"role": "user", "content": "<image>\\nList a handful of essential elements in this visual."}},
    {{"role": "assistant", "content": "There are two categories in the chart. The rate in Guyana is not twice that of Papua New Guinea."}}
  ]
}}
```
"""
    (dataset_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy the usable ureader image subset into a local dataset directory."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rows = load_rows(args.manifest)
    dataset_dir = args.output_dir
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"

    copy_jobs = []
    rewritten_rows = []
    for row in rows:
        src_rel = row["image"]
        src = args.source_image_root / src_rel
        dst_rel = Path("images") / src_rel
        dst = dataset_dir / dst_rel
        copy_jobs.append((src, dst))

        rewritten = dict(row)
        rewritten["image"] = dst_rel.as_posix()
        rewritten_rows.append(rewritten)

    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(copy_one, src, dst) for src, dst in copy_jobs]
        for future in as_completed(futures):
            _, size = future.result()
            total_bytes += size

    write_jsonl(rewritten_rows, annotations_dir / "train.jsonl")
    write_readme(
        dataset_dir, rewritten_rows, total_bytes, args.manifest, args.source_image_root
    )

    summary = {
        "dataset_dir": str(dataset_dir),
        "samples": len(rewritten_rows),
        "images_dir": str(images_dir),
        "annotations": str(annotations_dir / "train.jsonl"),
        "image_bytes": total_bytes,
        "image_gib": round(total_bytes / 1024 / 1024 / 1024, 3),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

