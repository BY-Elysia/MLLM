import argparse
from pathlib import Path

try:
    from data import CLIPJsonlDataset, build_clip_text
except ImportError:
    from .data import CLIPJsonlDataset, build_clip_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview several CLIP image-text pairs.")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("outputs/clip_preview_pairs.md"),
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--text-modes",
        nargs="+",
        default=["assistant", "qa", "assistant_with_question"],
    )
    return parser.parse_args()


def infer_group(image_path: str) -> str:
    parts = Path(image_path).parts
    if "docvqa" in parts:
        return "docvqa"
    if "ChartQA" in parts:
        return "ChartQA"
    if "InfographicsVQA" in parts:
        return "InfographicsVQA"
    if "VisualMRC" in parts:
        return "VisualMRC"
    if "DUE_Benchmark" in parts:
        return "DUE_Benchmark"
    return "other"


def select_diverse_samples(dataset: CLIPJsonlDataset, limit: int) -> list:
    selected = []
    seen_groups = set()

    for sample in dataset.records:
        group = infer_group(sample.image_path)
        if group in seen_groups:
            continue
        selected.append(sample)
        seen_groups.add(group)
        if len(selected) >= limit:
            return selected

    for sample in dataset.records:
        if len(selected) >= limit:
            break
        if sample in selected:
            continue
        selected.append(sample)

    return selected


def to_link(path: str) -> str:
    resolved = Path(path).resolve()
    return f"[{resolved.name}]({resolved})"


def build_markdown(dataset: CLIPJsonlDataset, samples: list, text_modes: list[str]) -> str:
    lines = [
        "# CLIP 图文对预览",
        "",
        f"- 标注文件：`{dataset.annotations_path}`",
        f"- 样本数量：`{len(dataset)}`",
        f"- 预览条数：`{len(samples)}`",
        "",
    ]

    for index, sample in enumerate(samples, start=1):
        record = sample.record
        lines.append(f"## 样本 {index}")
        lines.append("")
        lines.append(f"- id：`{sample.sample_id}`")
        lines.append(f"- 分组：`{infer_group(sample.image_path)}`")
        lines.append(f"- 图片：{to_link(sample.image_path)}")
        lines.append(f"- 原始 user：`{(record.get('user') or '').strip()}`")
        lines.append(f"- 原始 assistant：`{(record.get('assistant') or '').strip()}`")
        lines.append("")

        for mode in text_modes:
            text = build_clip_text(record, text_mode=mode)
            lines.append(f"### text_mode = `{mode}`")
            lines.append("")
            lines.append(text)
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    dataset = CLIPJsonlDataset(
        annotations_path=args.annotations,
        dataset_root=args.dataset_root,
        text_mode="assistant",
    )
    samples = select_diverse_samples(dataset, limit=args.limit)
    markdown = build_markdown(dataset, samples, text_modes=args.text_modes)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown, encoding="utf-8")
    print(args.output_markdown.resolve())


if __name__ == "__main__":
    main()
