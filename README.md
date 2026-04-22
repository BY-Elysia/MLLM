# MLLM

Minimal repository scaffold for preparing and fine-tuning a multimodal language model on the cleaned UReader subset.

## Remote

- GitHub: `https://github.com/BY-Elysia/MLLM.git`

## Current Scope

- Keep data-processing scripts under version control
- Keep datasets, checkpoints, and experiment outputs out of git
- Start from the available `4229` image-text samples that were already cleaned locally

## Repository Layout

```text
MLLM/
├── README.md
├── .gitignore
├── scripts/
│   ├── prepare_ureader_kg.py
│   └── organize_existing_subset.py
├── data/
├── datasets/
├── artifacts/
└── outputs/
```

## Included Scripts

- `scripts/prepare_ureader_kg.py`
  - Repairs malformed fragments in `ureader_kg_processed.json`
  - Exports JSONL
  - Optionally filters out samples whose images are missing
- `scripts/organize_existing_subset.py`
  - Copies the currently usable image subset into a local dataset directory
  - Rewrites annotation paths to the copied local images

## Recommended Workflow

1. Place the raw annotation JSON and image folders on local disk.
2. Run `prepare_ureader_kg.py` to create a cleaned JSONL manifest.
3. Run `organize_existing_subset.py` to copy the usable subset into a training-ready local directory.
4. Build the training loader for the target model family such as `Qwen2.5-VL`, `InternVL`, or `LLaVA`.

## Example Commands

Clean and export a JSONL manifest:

```bash
python3 scripts/prepare_ureader_kg.py \
  --input-json /path/to/ureader_kg_processed.json \
  --image-root /path/to/ureader_kg_images \
  --drop-missing-images \
  --output-jsonl outputs/ureader_existing_only.jsonl \
  --summary-json outputs/summary_existing_only.json
```

Organize the usable subset into a local dataset directory:

```bash
python3 scripts/organize_existing_subset.py \
  --manifest outputs/ureader_existing_only.jsonl \
  --source-image-root /path/to/ureader_kg_images \
  --output-dir datasets/ureader_existing_local
```

## Notes

- The repository is initialized for code and metadata, not for committing raw images or model weights.
- If you later want to train directly from this repo, the next sensible addition is a model-specific dataset loader and training entrypoint.

