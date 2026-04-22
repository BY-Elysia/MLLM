import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small JSONL subset for CLIP smoke tests.")
    parser.add_argument("--input", required=True, type=Path, help="Source JSONL annotations.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL annotations.")
    parser.add_argument("--limit", type=int, default=64, help="Number of samples to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def make_subset(
    input_path: str | Path,
    output_path: str | Path,
    limit: int = 64,
    seed: int = 42,
) -> dict[str, object]:
    input_path = Path(input_path)
    output_path = Path(output_path)

    rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(line)

    if not rows:
        raise RuntimeError(f"No records found in {input_path}")

    rng = random.Random(seed)
    rng.shuffle(rows)

    subset = rows[: min(limit, len(rows))]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.writelines(subset)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "limit": limit,
        "written": len(subset),
        "seed": seed,
    }


def main() -> None:
    args = parse_args()
    result = make_subset(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
