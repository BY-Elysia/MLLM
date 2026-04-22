# MLLM

这是一个用于多模态大模型数据整理与微调准备的基础仓库，当前以清洗后的 UReader 子集为起点。

## 远端仓库

- GitHub：`https://github.com/BY-Elysia/MLLM.git`

## 当前目标

- 将数据处理脚本纳入版本控制
- 不把原始数据、训练产物和模型权重提交到 git
- 先基于本地已清洗完成的 `4229` 条图文样本开展后续训练准备

## 仓库结构

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

## 已包含脚本

- `scripts/prepare_ureader_kg.py`
  - 修复 `ureader_kg_processed.json` 中损坏的片段
  - 导出清洗后的 `JSONL`
  - 可选过滤掉缺失图片的样本
- `scripts/organize_existing_subset.py`
  - 将当前可用图片子集复制到本地数据集目录
  - 将标注中的图片路径重写为复制后的本地相对路径

## 推荐流程

1. 将原始标注 `JSON` 和图片目录准备到本地磁盘。
2. 运行 `prepare_ureader_kg.py` 生成清洗后的 `JSONL` 清单。
3. 运行 `organize_existing_subset.py` 把可用子集整理成训练可读的本地目录。
4. 再按目标模型补上数据加载器和训练入口，例如 `Qwen2.5-VL`、`InternVL` 或 `LLaVA`。

## 示例命令

清洗并导出 `JSONL`：

```bash
python3 scripts/prepare_ureader_kg.py \
  --input-json /path/to/ureader_kg_processed.json \
  --image-root /path/to/ureader_kg_images \
  --drop-missing-images \
  --output-jsonl outputs/ureader_existing_only.jsonl \
  --summary-json outputs/summary_existing_only.json
```

整理当前可用子集到本地数据集目录：

```bash
python3 scripts/organize_existing_subset.py \
  --manifest outputs/ureader_existing_only.jsonl \
  --source-image-root /path/to/ureader_kg_images \
  --output-dir datasets/ureader_existing_local
```

## 说明

- 这个仓库当前用于管理代码和元数据，不用于提交原始图片或模型权重。
- 如果后续要直接在这个仓库里启动训练，下一步最合理的补充是模型专用的 `dataset loader` 和训练脚本。
