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


def main() -> None:
    args = parse_args()

    rows = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(line)

    if not rows:
        raise RuntimeError(f"No records found in {args.input}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    subset = rows[: min(args.limit, len(rows))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.writelines(subset)

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "limit": args.limit,
                "written": len(subset),
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
