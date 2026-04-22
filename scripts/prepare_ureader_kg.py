#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path


def split_top_level_objects(text: str) -> list[str]:
    items: list[str] = []
    in_string = False
    escaped = False
    depth = 0
    start = None

    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                items.append(text[start : idx + 1])
                start = None

    return items


def salvage_record_fragment(raw: str) -> dict | None:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    image_match = re.search(r'"image":\s*"([^"]+)"', raw)
    if not image_match:
        return None
    image_path = image_match.group(1)

    data_source_match = re.search(r'"data_source":\s*"([^"]+)"', raw)
    data_source = data_source_match.group(1) if data_source_match else "ureader_kg"

    conv_match = re.search(
        r'"conversations":\s*(\[\{.*\}\])\s*,\s*"data_source"', raw, re.S
    )
    repair_note = "recovered_fragment"
    if not conv_match:
        conv_match = re.search(
            r'^\{"id":\s*"[^"]*?:\s*(\[\{.*\}\])\s*,\s*"data_source"', raw, re.S
        )
        repair_note = "recovered_missing_conversations_key"
    if not conv_match:
        return None

    try:
        conversations = json.loads(conv_match.group(1))
    except json.JSONDecodeError:
        return None

    id_match = re.search(r'^\{"id":\s*"([^"]+)"', raw)
    raw_id = id_match.group(1) if id_match else ""
    record_id = raw_id
    if ":" in record_id or "[" in record_id or not record_id:
        record_id = Path(image_path).stem

    return {
        "id": record_id,
        "conversations": conversations,
        "data_source": data_source,
        "image": image_path,
        "_repair": repair_note,
    }


def try_repair_record(raw: str) -> list[dict]:
    starts = [match.start() for match in re.finditer(r'\{"id":\s*"', raw)]
    if not starts:
        repaired = salvage_record_fragment(raw)
        return [repaired] if repaired else []

    repaired_records: list[dict] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(raw)
        fragment = raw[start:end].rstrip(", \n")
        repaired = salvage_record_fragment(fragment)
        if repaired is not None:
            repaired_records.append(repaired)
    return repaired_records


def load_records(input_json: Path) -> tuple[list[dict], int, int]:
    text = input_json.read_text(encoding="utf-8")
    raw_items = split_top_level_objects(text)

    records: list[dict] = []
    repaired = 0
    invalid = 0

    for raw in raw_items:
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            repaired_records = try_repair_record(raw)
            if not repaired_records:
                invalid += 1
                continue
            repaired += len(repaired_records)
            records.extend(repaired_records)
            continue

        if isinstance(record, dict):
            records.append(record)
            continue

        invalid += 1

    return records, repaired, invalid


def build_existing_image_set(image_root: Path) -> set[str]:
    existing: set[str] = set()
    for dirpath, _, filenames in os.walk(image_root):
        for filename in filenames:
            rel = os.path.relpath(os.path.join(dirpath, filename), image_root)
            existing.add(rel.replace("\\", "/"))
    return existing


def normalize_record(
    record: dict, image_root: Path, absolute_image: bool, existing_images: set[str]
) -> dict:
    conversations = record.get("conversations") or []

    user_text = None
    assistant_text = None
    messages = []

    for turn in conversations:
        speaker = turn.get("from")
        text = turn.get("value", "")
        if speaker == "human" and user_text is None:
            user_text = text
            messages.append({"role": "user", "content": text})
        elif speaker == "gpt" and assistant_text is None:
            assistant_text = text
            messages.append({"role": "assistant", "content": text})

    image_rel = record.get("image", "")
    image_path = image_root / image_rel
    image_exists = image_rel in existing_images

    return {
        "id": record.get("id"),
        "data_source": record.get("data_source"),
        "image": str(image_path.resolve()) if absolute_image else image_rel,
        "user": user_text,
        "assistant": assistant_text,
        "messages": messages,
        "conversations": conversations,
        "image_exists": image_exists,
        "repair": record.get("_repair"),
    }


def write_jsonl(records: list[dict], output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and convert ureader_kg_processed.json into JSONL."
    )
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--drop-missing-images", action="store_true")
    parser.add_argument("--absolute-image", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    records, repaired, invalid = load_records(args.input_json)
    existing_images = build_existing_image_set(args.image_root)
    normalized = [
        normalize_record(record, args.image_root, args.absolute_image, existing_images)
        for record in records
    ]

    if args.drop_missing_images:
        normalized = [record for record in normalized if record["image_exists"]]

    if args.limit > 0:
        normalized = normalized[: args.limit]

    unique_images = {record["image"] for record in normalized}
    source_counts = Counter(record.get("data_source") for record in normalized)
    existing_count = sum(1 for record in normalized if record["image_exists"])

    summary = {
        "input_json": str(args.input_json),
        "image_root": str(args.image_root),
        "records_kept": len(normalized),
        "existing_image_records": existing_count,
        "missing_image_records": len(normalized) - existing_count,
        "unique_images": len(unique_images),
        "repaired_records": repaired,
        "invalid_records_dropped": invalid,
        "source_counts": dict(source_counts),
        "sample": normalized[0] if normalized else None,
    }

    if args.output_jsonl:
        write_jsonl(normalized, args.output_jsonl)
        summary["output_jsonl"] = str(args.output_jsonl)

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

