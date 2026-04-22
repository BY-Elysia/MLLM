# CLIP 流程说明

当前 `clip/` 目录只保留一套正式训练链路。

## 用法

1. 修改 [clip/config.json](/home/by/workspace/MLLM/clip/config.json:1)
2. 在仓库根目录运行：

```bash
python3 main.py
```

## 目录职责

- `clip/config.json`
  - CLIP 的唯一配置文件
- `clip/data.py`
  - 读取 `JSONL` 标注并构造图文训练样本
- `clip/model.py`
  - 基于 `transformers.CLIPModel` 的对比学习封装
- `clip/main.py`
  - 底层训练实现，负责数据集、训练循环、验证和 checkpoint

## 配置字段

主要字段如下：

- `train_annotations`
- `dataset_root`
- `val_annotations`
- `output_dir`
- `model_name`
- `text_mode`
- `epochs`
- `batch_size`
- `eval_batch_size`
- `num_workers`
- `learning_rate`
- `weight_decay`
- `val_ratio`
- `seed`
- `max_length`
- `log_interval`
- `save_every_epoch`
- `save_optimizer_state`
- `freeze_vision`
- `freeze_text`
- `freeze_projection`
- `freeze_logit_scale`
- `disable_amp`
- `device`

说明：

- 路径统一按仓库根目录解析
- `val_annotations=null` 时，会按 `val_ratio` 从训练集切分验证集
- `model_name` 可以是 Hugging Face 模型名，也可以是本地模型目录
- 默认只保存 `best` checkpoint，不按 epoch 全量落盘
- 默认不保存优化器状态，避免 `training_state.pt` 过大

## 数据准备

训练前仍然使用仓库里的两个整理脚本：

1. `scripts/prepare_ureader_kg.py`
2. `scripts/organize_existing_subset.py`

当前默认训练文件是：

```text
datasets/ureader_existing_local/annotations/train.jsonl
```

单条样本结构示例：

```json
{
  "id": "00326327001359",
  "image": "images/ureader-instruction-1.0/ChartQA/train/png/00326327001359.png",
  "user": "<image>\nList a handful of essential elements in this visual.",
  "assistant": "There are two categories in the chart. The rate in Guyana is not twice that of Papua New Guinea."
}
```

`clip/data.py` 会把它转换为：

- 图像：`image`
- 文本：按 `text_mode` 从 `user / assistant` 构造

## `text_mode`

可选值：

- `assistant`
- `user`
- `qa`
- `assistant_with_question`

当前建议先用 `assistant` 作为基线。

## 输出

输出目录由 `clip/config.json` 中的 `output_dir` 控制，例如：

```text
outputs/clip_full_run/
├── checkpoints/
└── run_summary.json
```

- `checkpoints/` 保存模型与训练状态
- `run_summary.json` 保存本次运行的参数与指标摘要

## 依赖

```bash
pip install torch torchvision transformers pillow
```

GPU 环境下，`torch` 需要安装和机器 CUDA 对应的构建。
