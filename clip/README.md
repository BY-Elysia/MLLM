# CLIP 流程说明

这份文档说明当前 `clip/` 目录下这套代码是怎么跑通的，以及从原始数据到 CLIP 训练的完整流程。

## 目录职责

- `clip/model.py`
  - 基于 `transformers.CLIPModel` 的对比学习封装
  - 提供图像编码、文本编码、相似度计算和对比学习损失
- `clip/data.py`
  - 读取 `JSONL` 标注
  - 解析图片路径
  - 把每条样本整理成 `image + text` 的 CLIP 训练对
  - 提供 `Dataset`、`collator` 和训练/验证切分
- `clip/main.py`
  - 训练入口
  - 支持单个标注文件随机划分训练/验证集
  - 也支持显式传入独立的验证集标注
  - 负责 checkpoint 和训练摘要输出
- `clip/configs/*.json`
  - 训练配置文件
  - 路径、模型、batch size、epoch、日志频率等都可以放在这里

## 整体数据流

当前这套 CLIP 训练流程是：

1. 准备原始数据
2. 用 `scripts/prepare_ureader_kg.py` 清洗损坏的原始 `JSON`
3. 用 `scripts/organize_existing_subset.py` 把当前可用图片整理成一个本地数据集目录
4. 用 `clip/data.py` 把本地 `train.jsonl` 变成 CLIP 图文对
5. 用 `clip/main.py` 训练 CLIP
6. 将模型权重、processor 和训练状态保存到 `output-dir/checkpoints`

可以把它理解成：

```text
原始 JSON + 图片目录
        ↓
prepare_ureader_kg.py
        ↓
清洗后的 JSONL
        ↓
organize_existing_subset.py
        ↓
本地数据集目录
  ├── images/
  └── annotations/train.jsonl
        ↓
clip/data.py
        ↓
CLIP 图文 batch
        ↓
clip/model.py + clip/main.py
        ↓
训练日志 / checkpoint / run_summary.json
```

## 依赖

至少需要这些依赖：

```bash
pip install torch torchvision transformers pillow
```

如果你用 GPU，`torch` 需要按你的 CUDA 版本安装对应构建。

## 第一步：清洗原始标注

原始数据如果还是：

- `ureader_kg_processed.json`
- `ureader_kg_images_xxx/`

先执行清洗：

```bash
python3 scripts/prepare_ureader_kg.py \
  --input-json /path/to/ureader_kg_processed.json \
  --image-root /path/to/ureader_kg_images \
  --drop-missing-images \
  --output-jsonl outputs/ureader_existing_only.jsonl \
  --summary-json outputs/summary_existing_only.json
```

这一步会做几件事：

- 尝试修复损坏的 JSON 片段
- 把原始标注转成逐行的 `JSONL`
- 如果加了 `--drop-missing-images`，会过滤掉图还不存在的样本

输出结果是一份 CLIP 和其他模型都容易接入的 `JSONL`。

## 第二步：整理成本地数据集

为了避免后续训练依赖外部盘符路径，建议再执行一次整理：

```bash
python3 scripts/organize_existing_subset.py \
  --manifest outputs/ureader_existing_only.jsonl \
  --source-image-root /path/to/ureader_kg_images \
  --output-dir datasets/ureader_existing_local
```

整理后的目录结构大致是：

```text
datasets/ureader_existing_local/
├── images/
│   └── ...
├── annotations/
│   └── train.jsonl
└── README.md
```

这一步会：

- 复制当前可用图片到本地目录
- 重写 `train.jsonl` 里的 `image` 字段
- 让训练阶段不再依赖原始数据所在位置

## 第三步：理解 `train.jsonl`

当前 CLIP 数据流默认读取这种结构：

```json
{
  "id": "00326327001359",
  "image": "images/ureader-instruction-1.0/ChartQA/train/png/00326327001359.png",
  "user": "<image>\nList a handful of essential elements in this visual.",
  "assistant": "There are two categories in the chart. The rate in Guyana is not twice that of Papua New Guinea."
}
```

在 `clip/data.py` 里，会把这条数据转换为一个 CLIP 样本：

- 图像：`image`
- 文本：根据 `text_mode` 从 `user / assistant` 中构造

## 第四步：选择 `text_mode`

`clip/main.py` 支持这些文本构造方式：

- `assistant`
  - 只拿答案文本做图文配对
  - 最接近“图片 -> 解释/描述”
- `user`
  - 只拿问题文本做图文配对
  - 更偏“图片 -> 问题语义对齐”
- `qa`
  - 把问题和答案拼起来
  - 适合保留更多语义信息
- `assistant_with_question`
  - 以 `Question: ... / Answer: ...` 的形式拼接
  - 比简单拼接更明确

如果你只是先验证 CLIP 能不能跑通，建议先从 `assistant` 开始。

## 第五步：启动训练

在仓库根目录执行。现在推荐优先使用配置文件：

```bash
python3 -u -m clip.main --config clip/configs/small_demo.json
```

也可以继续用命令行参数覆盖配置文件里的值：

```bash
python3 -u -m clip.main \
  --config clip/configs/small_demo.json \
  --epochs 3 \
  --batch-size 16
```

完整命令行方式依然可用：

```bash
python3 -m clip.main \
  --train-annotations datasets/ureader_existing_local/annotations/train.jsonl \
  --dataset-root datasets/ureader_existing_local \
  --output-dir outputs/clip_run \
  --model-name openai/clip-vit-base-patch32 \
  --text-mode assistant \
  --epochs 5 \
  --batch-size 32 \
  --val-ratio 0.1
```

如果你已经准备好了单独的验证集，也可以这样：

```bash
python3 -m clip.main \
  --train-annotations /path/to/train.jsonl \
  --val-annotations /path/to/val.jsonl \
  --dataset-root /path/to/dataset_root \
  --output-dir outputs/clip_run
```

## 第六步：训练入口会做什么

`clip/main.py` 的流程是：

1. 读取命令行参数
2. 如果提供了 `--config`，先加载 JSON 配置
3. 加载 `AutoProcessor`
4. 构造 `CLIPJsonlDataset`
5. 如果没有显式验证集，就按 `val-ratio` 随机切分
6. 构造 `DataLoader`
7. 加载 `CLIPContrastiveModel`
8. 建立优化器 `AdamW`
9. 用余弦退火学习率调度器训练
10. 每个 epoch 输出 loss / image_acc / text_acc
11. 保存 checkpoint 和训练摘要

## 训练输出

默认输出目录是：

```text
outputs/clip/
```

如果你用 `--output-dir outputs/clip_run`，则会生成类似：

```text
outputs/clip_run/
├── checkpoints/
│   ├── best/
│   │   ├── clip/
│   │   ├── processor/
│   │   └── training_state.pt
│   ├── epoch-001/
│   └── ...
└── run_summary.json
```

其中：

- `clip/`
  - Hugging Face 风格保存的 CLIP 权重
- `processor/`
  - 对应的图像与文本预处理器
- `training_state.pt`
  - 训练状态，包括优化器、调度器和当前指标
- `run_summary.json`
  - 本次运行的参数和最佳指标摘要

## 常用参数

常用训练参数：

- `--epochs`
- `--batch-size`
- `--eval-batch-size`
- `--learning-rate`
- `--weight-decay`
- `--val-ratio`
- `--max-length`
- `--device`

冻结相关参数：

- `--freeze-vision`
- `--freeze-text`
- `--freeze-projection`
- `--freeze-logit-scale`

其他控制参数：

- `--save-every-epoch`
- `--disable-amp`

## 路径解析规则

`clip/data.py` 解析图片路径时遵循这套规则：

- 如果 `image` 已经是绝对路径，直接使用
- 如果传了 `--dataset-root`，则拼成 `dataset_root / image`
- 如果没传 `--dataset-root`，则拼成 `annotations_path.parent / image`

所以最稳妥的方式是：

- 标注里的 `image` 写相对路径
- 启动训练时显式传 `--dataset-root`

## 当前更适合的用途

你现在这份数据更适合拿来做：

- 图文检索
- 图文对齐
- 图像和文本嵌入空间对比学习

它不等价于“多模态指令微调”。

因为 CLIP 的目标是：

- `image -> embedding`
- `text -> embedding`
- 让匹配的图文更接近，不匹配的更远

而不是直接生成答案文本。

## 训练后怎么用

训练完成后，你可以用 `clip/model.py` 里的接口做两类事：

- `encode_image(...)`
  - 把图片编码成向量
- `encode_text(...)`
  - 把文本编码成向量

然后做：

- 图搜文
- 文搜图
- 图文相似度排序
- 检索召回前置模块

## 一个建议

如果你只是先把工程打通，建议先用这组参数：

```bash
python3 -m clip.main \
  --train-annotations datasets/ureader_existing_local/annotations/train.jsonl \
  --dataset-root datasets/ureader_existing_local \
  --output-dir outputs/clip_debug \
  --text-mode assistant \
  --epochs 1 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --val-ratio 0.1
```

先确认：

- 数据能正常读取
- batch 能正常构造
- loss 能下降
- checkpoint 能正常写出

然后再放大 batch、epoch 和模型规模。
