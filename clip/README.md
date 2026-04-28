# CLIP 训推流程说明

这份文档说明当前仓库里 `clip/` 这套代码怎么训练、怎么保存、怎么做推理。

当前入口已经固定成一条链路：

1. 修改 [clip/config.json](/home/by/workspace/MLLM/clip/config.json:1)
2. 在仓库根目录运行 `python3 main.py`

## 1. 目录职责

- `main.py`
  - 仓库根入口
  - 读取 `clip/config.json`
  - 统一把相对路径按仓库根目录解析
- `clip/config.json`
  - 当前 CLIP 的唯一配置文件
- `clip/data.py`
  - 读取 `JSONL`
  - 解析图片路径
  - 构造图文对
  - 切分训练集和验证集
- `clip/model.py`
  - 基于 `transformers.CLIPModel` 的训练封装
  - 提供 `encode_image`、`encode_text`、`compute_similarity`
- `clip/main.py`
  - 底层训练实现
  - 负责 dataloader、训练循环、验证、checkpoint、summary

## 2. 这套代码在做什么

这不是生成式多模态指令微调，而是标准 CLIP 式图文对比学习。

训练目标是：

- 图像编码成向量
- 文本编码成向量
- 匹配图文更接近
- 不匹配图文更远

当前损失是双向对比学习 loss：

- `image -> text`
- `text -> image`

当前日志里的：

- `loss`
- `img_acc`
- `text_acc`

含义是 batch 内的 top-1 检索准确率，不是全库检索指标。

## 3. 环境准备

最少依赖：

```bash
pip install torch torchvision transformers pillow
```

如果你用 GPU，`torch` 需要安装和机器 CUDA 匹配的版本。

环境验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 4. 数据准备

当前训练默认读取：

```text
datasets/ureader_existing_local/annotations/train.jsonl
```

如果你还没把原始数据整理成这个格式，先跑仓库里的两个脚本：

1. `scripts/prepare_ureader_kg.py`
2. `scripts/organize_existing_subset.py`

整理后的目录通常是：

```text
datasets/ureader_existing_local/
├── annotations/
│   └── train.jsonl
└── images/
    └── ...
```

单条标注示例：

```json
{
  "id": "00326327001359",
  "image": "images/ureader-instruction-1.0/ChartQA/train/png/00326327001359.png",
  "user": "<image>\nList a handful of essential elements in this visual.",
  "assistant": "There are two categories in the chart. The rate in Guyana is not twice that of Papua New Guinea."
}
```

训练时使用的字段：

- `image`
  - 图片路径
- `user`
  - 问题文本
- `assistant`
  - 答案文本

`clip/data.py` 会把它转成 `CLIPSample`：

- `sample_id`
- `image_path`
- `text`
- `record`

## 5. 文本构造方式

`clip/data.py` 目前支持 4 种 `text_mode`：

- `assistant`
  - 只使用答案文本
  - 最适合当前这批“图片 + 问答答案”数据做第一版基线
- `user`
  - 只使用问题文本
- `qa`
  - 问题和答案直接拼接
- `assistant_with_question`
  - 格式化为 `Question: ... / Answer: ...`

当前建议先用：

```json
"text_mode": "assistant"
```

## 6. 配置文件怎么改

当前唯一配置文件是 [clip/config.json](/home/by/workspace/MLLM/clip/config.json:1)。

默认内容类似：

```json
{
  "train_annotations": "datasets/ureader_existing_local/annotations/train.jsonl",
  "dataset_root": "datasets/ureader_existing_local",
  "val_annotations": null,
  "output_dir": "outputs/clip_full_run",
  "model_name": "models/clip-vit-base-patch32",
  "text_mode": "assistant",
  "epochs": 5,
  "batch_size": 16,
  "eval_batch_size": 16,
  "num_workers": 4,
  "learning_rate": 5e-05,
  "weight_decay": 0.01,
  "val_ratio": 0.2,
  "seed": 42,
  "max_length": 77,
  "log_interval": 20,
  "save_every_epoch": false,
  "save_optimizer_state": false,
  "freeze_vision": false,
  "freeze_text": false,
  "freeze_projection": false,
  "freeze_logit_scale": false,
  "disable_amp": false,
  "device": null
}
```

关键字段说明：

- `train_annotations`
  - 训练标注文件
- `dataset_root`
  - 用来拼接相对图片路径
- `val_annotations`
  - 如果为 `null`，就从训练集里按 `val_ratio` 切分
- `output_dir`
  - 训练输出目录
- `model_name`
  - Hugging Face 模型名，或者本地模型目录
- `epochs`
  - 训练 epoch 数
- `batch_size`
  - 训练 batch size
- `eval_batch_size`
  - 验证 batch size
- `num_workers`
  - dataloader worker 数
- `learning_rate`
  - 学习率
- `weight_decay`
  - 权重衰减
- `val_ratio`
  - 验证集比例
- `max_length`
  - 文本 token 最大长度
- `log_interval`
  - 每隔多少 step 打一次训练日志
- `save_every_epoch`
  - 是否每个 epoch 保存一次 checkpoint
- `save_optimizer_state`
  - 是否把 optimizer/scheduler 状态写到 `training_state.pt`
- `freeze_*`
  - 控制是否冻结视觉塔、文本塔、投影层、logit scale
- `disable_amp`
  - 是否关闭混合精度
- `device`
  - 手动指定设备，例如 `cpu`、`cuda`

当前默认保存策略是：

- `save_every_epoch = false`
- `save_optimizer_state = false`

这是为了避免 checkpoint 太大，把训练中途写盘写崩。

## 7. 训练怎么跑

在仓库根目录执行：

```bash
python3 main.py
```

如果你要显式指定别的配置文件，也可以：

```bash
python3 main.py --config clip/config.json
```

## 8. 按当前默认配置的具体训练规模

如果使用你当前这份整理好的全量数据，也就是之前那份 `ureader_existing_local`，总样本量是：

- 总样本数：`4229`
- 当前 `val_ratio`：`0.2`
- 验证集：`845`
- 训练集：`3384`

配合当前默认配置：

```json
"batch_size": 16,
"eval_batch_size": 16,
"max_length": 77,
"model_name": "models/clip-vit-base-patch32"
```

训练和验证的 step 数大致是：

- 训练 step/epoch：`ceil(3384 / 16) = 212`
- 验证 step/epoch：`ceil(845 / 16) = 53`

这也就是你日志里常见的：

```text
step=20/212
```

### 8.1 每个 batch 里的主要张量大小

以当前默认 `batch_size = 16` 为例，训练时进入模型的主要张量通常是：

- `pixel_values`
  - shape: `[16, 3, 224, 224]`
  - 含义：`16` 张 RGB 图像
- `input_ids`
  - shape: `[16, L]`，其中 `L <= 77`
  - 含义：当前 batch 中文本 token 化后的实际长度，受 `77` 的上限约束
- `attention_mask`
  - shape: `[16, L]`
  - 含义：文本有效 token 掩码

模型前向后的主要张量：

- `image_embeds`
  - shape: `[16, 512]`
- `text_embeds`
  - shape: `[16, 512]`
- `logits_per_image`
  - shape: `[16, 16]`
- `logits_per_text`
  - shape: `[16, 16]`

解释：

- 最终图像向量和文本向量都被投影到 `512` 维
- 相似度矩阵是一个 `batch_size x batch_size` 的方阵
- CLIP 的正样本在这个矩阵的对角线上
- 文本长度 `77` 在当前实现里是上限，不是每条样本都必须正好等于 `77`

最后一个 batch 可能小于 `16`。例如：

- 训练集最后一个 batch：`3384 % 16 = 8`
  - `pixel_values` 会变成 `[8, 3, 224, 224]`
  - `logits_per_image` 会变成 `[8, 8]`
- 验证集最后一个 batch：`845 % 16 = 13`
  - `pixel_values` 会变成 `[13, 3, 224, 224]`
  - `logits_per_image` 会变成 `[13, 13]`

### 8.2 当前模型的具体规格

当前配置里的：

```json
"model_name": "models/clip-vit-base-patch32"
```

如果这个目录对应的是官方 `openai/clip-vit-base-patch32`，那它的关键规格可以按下面理解：

- 模型类型：`CLIP ViT-B/32`
- 输入图像尺寸：`224 x 224`
- 图像通道：`3`
- patch size：`32`
- 视觉塔隐藏维度：`768`
- 文本最大长度：`77`
- 最终图文共享嵌入维度：`512`

这也是为什么当前训练里的核心 shape 是：

- 图像输入：`[B, 3, 224, 224]`
- 文本输入：`[B, 77]`
- 图像向量：`[B, 512]`
- 文本向量：`[B, 512]`
- 相似度矩阵：`[B, B]`

### 8.3 视觉编码的具体过程

当前视觉编码在代码里的调用链是：

```text
CLIPBatchCollator
  -> processor(images=..., return_tensors="pt")
  -> pixel_values
  -> CLIPContrastiveModel.encode_image(...)
  -> clip.get_image_features(...)
  -> visual_projection
  -> normalize
```

对应到当前默认模型 `CLIP ViT-B/32`，可以按下面理解。

#### 第一步：图像预处理

`clip/data.py` 里的 `CLIPBatchCollator` 会先：

1. 读取图片
2. 转成 `RGB`
3. 交给 `AutoProcessor`

processor 会做典型的 CLIP 图像预处理：

- resize
- center crop
- normalize
- 转成 tensor

所以进入模型前的图像张量是：

- `pixel_values`
  - shape: `[B, 3, 224, 224]`

例如默认 batch size 为 16 时：

- `pixel_values = [16, 3, 224, 224]`

#### 第二步：patch embedding

对 `224 x 224` 图像，`patch_size = 32` 时：

- 高方向 patch 数：`224 / 32 = 7`
- 宽方向 patch 数：`224 / 32 = 7`
- 总 patch 数：`7 x 7 = 49`

视觉塔首先会把图像切成 patch，并映射到视觉隐藏维：

- patch embedding 后大致可以理解为：`[B, 49, 768]`

如果按卷积实现的中间视角看，也可以理解成：

- `[B, 768, 7, 7]`

然后再展平为：

- `[B, 49, 768]`

#### 第三步：加入 class token 和位置编码

ViT 会在 patch token 前面拼一个全局 token：

- class token 数量：`1`
- patch token 数量：`49`

所以送入视觉 Transformer 的 token 序列长度是：

- `1 + 49 = 50`

对应张量可以理解为：

- `[B, 50, 768]`

这一步还会加上 position embedding。

#### 第四步：视觉 Transformer 编码

视觉 Transformer 会在这个序列上做多层 self-attention 编码。

编码后输出仍然可以理解成：

- `[B, 50, 768]`

其中：

- 第 0 个 token 对应全局 class token
- 后面 49 个 token 对应图像 patch

#### 第五步：池化和投影

视觉塔最终会取全局表示，得到一个 pooled 特征：

- `pooler_output`
  - shape: `[B, 768]`

然后经过 `visual_projection` 线性投影到共享对比空间：

- 投影后：
  - shape: `[B, 512]`

在我们自己的封装里，这一步发生在：

- `clip/model.py`
- `encode_image(...)`
- `_coerce_feature_output(...)`
- `_project_if_needed(...)`

#### 第六步：归一化

最后 `encode_image(...)` 会做 `L2 normalize`：

- 最终图像向量：
  - shape: `[B, 512]`

这就是后面拿来和文本向量做点积相似度的 image embedding。

### 8.4 文本编码的具体过程

当前文本编码在代码里的调用链是：

```text
build_clip_text(...)
  -> processor(text=..., padding=True, truncation=True)
  -> input_ids / attention_mask
  -> CLIPContrastiveModel.encode_text(...)
  -> clip.get_text_features(...)
  -> text_projection
  -> normalize
```

#### 第一步：文本构造

在真正 tokenizer 之前，`clip/data.py` 会先根据 `text_mode` 构造文本。

比如默认：

```json
"text_mode": "assistant"
```

那就是优先取每条样本的 `assistant` 字段。

这一步的输出还是 Python 字符串列表，比如：

- `["There are two categories in the chart.", "...", ...]`

#### 第二步：tokenization

processor 会把文本转成 tokenizer 可用输入：

- `input_ids`
- `attention_mask`

当前默认：

```json
"max_length": 77
```

所以进入模型前常见的文本张量是：

- `input_ids`
  - shape: `[B, L]`，其中 `L <= 77`
- `attention_mask`
  - shape: `[B, L]`

这里的 `77` 是 CLIP 官方文本长度上限，不是“每条文本固定 77 个真实 token”。

在当前实现里，`clip/data.py` 调用 processor 的方式是：

```python
processor(
    text=texts,
    images=images,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=77,
)
```

这意味着：

- 如果某条文本很短，不会凭空变出 `77` 个真实 token
- 它会先正常分词
- 然后在当前 batch 内按需要补 pad token
- 超过 `77` 的部分会被截断

所以更准确地说：

- `77` 是长度上限
- 当前 batch 的实际张量宽度是 `L`
- `L` 不会超过 `77`

如果你把 `padding=True` 改成 `padding="max_length"`，那文本张量才会固定成：

- `input_ids: [B, 77]`
- `attention_mask: [B, 77]`

#### 第三步：token embedding 和位置编码

文本塔先做：

1. token embedding lookup
2. position embedding

对于 `ViT-B/32` 对应的官方 CLIP 文本塔，可以把隐藏维理解为：

- 文本 hidden size：`512`

所以文本进入 Transformer 之前，大致可以理解为：

- `[B, 77, 512]`

#### 第四步：文本 Transformer 编码

文本塔会在长度为 `77` 的 token 序列上做 Transformer 编码。

编码后的张量仍然可以理解为：

- `[B, 77, 512]`

这里每个位置都有一个 contextualized token representation。

#### 第五步：取全局文本表示

CLIP 文本塔不会把所有 token 直接平均，而是从整句 token 序列里抽出一个“代表整句语义的单个向量”。

这就是这里说的 pooled feature。它的意思不是：

- 把所有 token 都保留下来继续做对比学习
- 或者简单把所有 token 平均一下

而是：

- 把整句压成一个固定长度的句向量
- 让这一个向量代表整句语义

对 CLIP 文本塔来说，源码里的真实行为更具体一些：

- 文本经过 Transformer 后，每个 token 都有一个输出向量
- 然后取出一个特定位置的向量作为句级表示
- 再把它映射到共享图文空间

如果按我们当前这套代码实际调用的 Hugging Face `CLIPModel / CLIPTextModel` 来说，精确行为是：

1. 先得到 `last_hidden_state`
   - shape: `[B, L, hidden_size]`
2. 再根据 `input_ids` 找到要取的那个位置
3. 用这个位置对应的 hidden state 作为 `pooler_output`
4. 最后再过 `text_projection`

这里“要取的那个位置”在 Hugging Face 当前实现里分两种情况：

- 如果 `eos_token_id == 2`
  - 走兼容旧配置的分支
  - 用 `input_ids.argmax(dim=-1)` 找位置
  - 这个写法依赖一个前提：`EOT token` 的 token id 在序列里是最大的
- 否则
  - 走新分支
  - 直接找 `input_ids == eos_token_id` 的第一个位置
  - 然后取那个位置对应的 hidden state

所以对“官方 CLIP tokenizer + 正常输入”来说，说“取 EOT 对应的最终隐藏状态”是对的；  
我之前说“近似理解”为了避免把不同实现分支混成一句话，但如果你问“实际到底是什么”，那答案就是：

- OpenAI 原版 CLIP：直接取 `text.argmax(dim=-1)` 对应位置的 hidden state，再乘 `text_projection`
- Hugging Face 当前 CLIP：
  - 旧兼容分支：也是 `argmax(input_ids)` 选位置
  - 新分支：显式找 `eos_token_id` 的位置

在我们当前这份工程里，因为 `clip/model.py` 调的是：

- `self.clip.get_text_features(...)`

而 `get_text_features(...)` 内部又调用 `self.text_model(...)`，再取 `pooler_output`，所以这里真正用到的是 Hugging Face 这套逻辑，而不是我们自己手写 pooling。

最终这一步得到的可以理解为：

- 文本 pooled 输出
  - shape: `[B, 512]`

这一步在我们代码里是通过：

- `clip.get_text_features(...)`
- 或 `_coerce_feature_output(...)` 兼容不同返回格式

拿到的。

你可以把它和 token 级输出区分开来看：

- token 级输出：`[B, L, 512]`
  - 每个 token 一个向量
- 句级 pooled 特征：`[B, 512]`
  - 整句只有一个向量

当前 CLIP 做图文对比学习时，用的是后者。

#### 第六步：投影和归一化

然后文本特征会经过 `text_projection`，进入和图像同一个共享空间。

对于这个 base 模型，文本隐藏维和投影维都落在 `512`，所以这里张量大小通常还是：

- `[B, 512]`

最后 `encode_text(...)` 同样会做 `L2 normalize`：

- 最终文本向量：
  - shape: `[B, 512]`

### 8.5 图文相似度矩阵是怎么来的

当 `image_embeds` 和 `text_embeds` 都准备好之后，`clip/model.py` 会做：

```text
logits_per_image = logit_scale * image_embeds @ text_embeds.T
logits_per_text  = logits_per_image.T
```

如果当前 batch size 是 `16`，那么：

- `image_embeds`: `[16, 512]`
- `text_embeds`: `[16, 512]`
- `text_embeds.T`: `[512, 16]`
- `logits_per_image`: `[16, 16]`

这个 `16 x 16` 矩阵里：

- 第 `i` 行表示第 `i` 张图和所有文本的相似度
- 第 `j` 列表示所有图和第 `j` 条文本的相似度
- 对角线位置对应配对正确的正样本

后面的对比学习 loss 就是让对角线更大，非对角线更小。

### 8.6 当前默认配置下的资源直觉

按 `batch_size = 16`、`ViT-B/32`、`224x224` 图像来跑，这属于比较温和的一档 CLIP 训练配置。

一般直觉上：

- `batch_size` 翻倍
  - `pixel_values` 的 batch 维翻倍
  - 相似度矩阵从 `[B, B]` 变成更大的方阵
  - 显存和计算都会明显增加
- `max_length` 增大
  - 文本张量从 `[B, 77]` 变长
  - 文本塔计算量增加
- 换更大的 backbone
  - 视觉塔和文本塔参数量、显存占用都会增加

## 9. 训练时内部实际发生了什么

执行 `python3 main.py` 之后，流程是：

1. 根目录 `main.py` 读取 `clip/config.json`
2. 把相对路径都按仓库根目录解析
3. 调用 `clip.main.run_training`
4. `AutoProcessor.from_pretrained(model_name)` 加载 processor
5. `CLIPJsonlDataset` 读取 `train.jsonl`
6. 如果没有 `val_annotations`，按 `val_ratio` 随机切分验证集
7. `CLIPBatchCollator` 负责：
   - 打开图片
   - 转 RGB
   - 调用 processor 做图像和文本编码
8. `CLIPContrastiveModel.from_pretrained(model_name)` 加载 CLIP
9. 建立：
   - `AdamW`
   - `CosineAnnealingLR`
10. 训练时每个 batch 做：
   - 图像前向
   - 文本前向
   - 相似度矩阵
   - 对比损失
   - 反向传播
   - optimizer step
11. 每个 epoch 后做验证
12. 如果验证 loss 更好，就保存 `best` checkpoint
13. 最后写 `run_summary.json`

## 10. 训练日志怎么看

训练日志示例：

```text
[2026-04-22 16:07:03] Starting epoch 1/5
[Epoch 1/5] step=20/212 loss=1.2345 img_acc=0.5625 text_acc=0.5781 time=6.9s
[Eval 1/5] loss=0.9876 img_acc=0.6250 text_acc=0.6012
```

解释：

- `step=20/212`
  - 当前 epoch 第 20 个 step，总共 212 个 step
- `loss`
  - 当前 epoch 到这个 step 的平均 loss
- `img_acc`
  - 以 image 为 query，在当前 batch 内检索正确 text 的 top-1 准确率
- `text_acc`
  - 以 text 为 query，在当前 batch 内检索正确 image 的 top-1 准确率
- `time`
  - 当前 epoch 已用时间
- `Eval ...`
  - 当前 epoch 验证集上的指标

通常你应该关注：

- 训练 `loss` 是否整体下降
- 验证 `loss` 是否下降
- 训练 `img_acc / text_acc` 是否上升
- 验证指标是否开始恶化

## 11. 输出目录长什么样

假设：

```json
"output_dir": "outputs/clip_full_run"
```

训练后目录通常是：

```text
outputs/clip_full_run/
├── checkpoints/
│   └── best/
│       ├── clip/
│       ├── processor/
│       └── training_state.pt
└── run_summary.json
```

说明：

- `checkpoints/best/clip/`
  - Hugging Face 格式保存的模型权重
- `checkpoints/best/processor/`
  - 对应的 processor
- `training_state.pt`
  - 训练状态摘要
  - 默认不含重复模型权重
  - 默认不含优化器状态
- `run_summary.json`
  - 本次运行的参数和最优指标摘要

如果你把：

```json
"save_every_epoch": true
```

打开，就会看到：

```text
checkpoints/epoch-001/
checkpoints/epoch-002/
...
```

但这会显著增加磁盘写入和存储占用。

## 12. 推理怎么做

当前这套 CLIP 推理，本质就是：

1. 读取训练好的 `clip/` 权重
2. 读取对应 `processor/`
3. 编码图像和文本
4. 计算相似度

### 12.1 单张图片和单条文本算相似度

```python
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from clip.model import CLIPContrastiveModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_root = Path("outputs/clip_full_run/checkpoints/best")

processor = AutoProcessor.from_pretrained(checkpoint_root / "processor")
model = CLIPContrastiveModel.from_pretrained(checkpoint_root / "clip").to(device)
model.eval()

image = Image.open("datasets/ureader_existing_local/images/xxx.png").convert("RGB")
text = "There are two categories in the chart."

batch = processor(
    text=[text],
    images=[image],
    return_tensors="pt",
    padding=True,
    truncation=True,
)
batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

with torch.no_grad():
    image_embeds = model.encode_image(batch["pixel_values"])
    text_embeds = model.encode_text(
        batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
    )
    logits_per_image, logits_per_text, logit_scale = model.compute_similarity(
        image_embeds=image_embeds,
        text_embeds=text_embeds,
    )

print("similarity =", logits_per_image[0, 0].item())
print("logit_scale =", logit_scale.item())
```

### 12.2 单张图片在多条候选文本里选最匹配的一条

```python
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from clip.model import CLIPContrastiveModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_root = Path("outputs/clip_full_run/checkpoints/best")

processor = AutoProcessor.from_pretrained(checkpoint_root / "processor")
model = CLIPContrastiveModel.from_pretrained(checkpoint_root / "clip").to(device)
model.eval()

image = Image.open("datasets/ureader_existing_local/images/xxx.png").convert("RGB")
texts = [
    "A bar chart comparing two countries.",
    "A document page with multiple paragraphs.",
    "A screenshot of a government website.",
]

image_inputs = processor(images=[image], return_tensors="pt")
text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)

image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

with torch.no_grad():
    image_embeds = model.encode_image(image_inputs["pixel_values"])
    text_embeds = model.encode_text(
        text_inputs["input_ids"],
        attention_mask=text_inputs.get("attention_mask"),
    )
    logits_per_image, _, _ = model.compute_similarity(image_embeds, text_embeds)

best_index = logits_per_image[0].argmax().item()
print("best_text =", texts[best_index])
print("scores =", logits_per_image[0].tolist())
```

### 12.3 批量做图搜文 / 文搜图

如果你要做批量检索，通常流程是：

1. 遍历图像库，提前算好全部 image embeddings
2. 遍历文本库，提前算好全部 text embeddings
3. 用矩阵乘法算相似度
4. 做 top-k 排序

也就是：

```text
image_embeds @ text_embeds.T
```

当前 `encode_image` 和 `encode_text` 默认已经做了归一化，适合直接拿来比相似度。

## 13. 一个推荐的正式训练流程

如果你要跑一版完整实验，建议是：

1. 确认本地模型路径可用
2. 确认 `clip/config.json` 中：
   - `train_annotations`
   - `dataset_root`
   - `output_dir`
   - `model_name`
3. 初次实验先用：
   - `text_mode = assistant`
   - `save_every_epoch = false`
   - `save_optimizer_state = false`
4. 跑：

```bash
python3 main.py
```

5. 看：
   - 训练日志
   - `run_summary.json`
   - `checkpoints/best/`

## 14. 当前实现的边界

这套代码现在已经能稳定完成单机 CLIP 训练和推理，但还没有这些能力：

- 断点续训
- 多卡分布式训练
- 更复杂的评测脚本
- 专门的批量 embedding 导出脚本
- 更高级的 hard negative 采样

所以当前最适合的用途是：

- 做第一版 CLIP 基线
- 做图文对齐实验
- 做图文检索前置模块

如果后面你要继续往正式实验推进，下一步最值得补的是：

- resume 训练
- 独立 inference/export 脚本
- 更完整的 retrieval evaluation
