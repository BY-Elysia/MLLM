# BLIP-2 训推流程说明

这份文档说明当前仓库里 `blip2/` 这套代码怎么训练、怎么保存，以及它现在相对原来那版 retrieval-only 实现做了哪些变化。

当前入口已经固定成一条链路：

1. 修改配置文件
2. 在仓库根目录运行 `python3 main.py --config ...`

当前提供两套配置：

1. [blip2/config.json](/home/by/workspace/MLLM/blip2/config.json:1)
   - 默认走 `stage1`
   - 更接近论文第一阶段
2. [blip2/stage2_config.json](/home/by/workspace/MLLM/blip2/stage2_config.json:1)
   - 走 `stage2`
   - 把 `Q-Former` 接到 LLM 上做生成式训练

## 1. 目录职责

- `main.py`
  - 仓库根入口
  - 读取指定配置文件
  - 根据 `model_family` 自动分发到 `clip` 或 `blip2`
- `blip2/config.json`
  - `stage1` 默认配置
- `blip2/stage2_config.json`
  - `stage2` 示例配置
- `clip/data.py`
  - `blip2` 直接复用这份数据层实现
  - 读取 `JSONL`
  - 解析图片路径
  - 构造样本
  - 切分训练集和验证集
- `blip2/model.py`
  - 现在包含两类模型封装：
  - `BLIP2Stage1Model`
    - 基于 `Blip2ForImageTextRetrieval`
    - 补齐更接近论文第一阶段的 3 个训练目标
  - `BLIP2Stage2Model`
    - 基于 `Blip2ForConditionalGeneration`
    - 负责把 `Q-Former` 输出接到 LLM
- `blip2/main.py`
  - 底层训练实现
  - 根据 `training_stage` 切换 dataloader、collator、训练循环和指标

## 2. 这套代码现在在做什么

这里先把“原论文 BLIP-2”和“当前仓库实现”分开。

原论文里的 BLIP-2 是：

- 两阶段预训练
- 第一阶段：
  - frozen image encoder + trainable Q-Former
  - 3 个目标联合训练：
    - image-text contrastive
    - image-text matching
    - image-grounded text generation
- 第二阶段：
  - frozen image encoder + Q-Former + frozen LLM
  - 把 `Q-Former` 输出作为 visual prefix / soft prompt 接给 LLM

当前仓库里的 `blip2/` 现在支持两条训练路径：

### 2.1 `stage1`

目标是尽量贴近论文第一阶段。

当前实现包含：

- `ITC`
  - image-text contrastive
- `ITM`
  - image-text matching
- `ITG`
  - image-grounded text generation

实现方式是：

- backbone 用 Hugging Face 的 `Blip2ForImageTextRetrieval`
- 复用它的：
  - `vision_model`
  - `query_tokens`
  - `qformer`
  - `vision_projection`
  - `text_projection`
  - `itm_head`
- 在仓库里额外补了一个 `Q-Former` 文本生成头
  - 用来做 `ITG`

所以它比之前那版只做 retrieval 的实现更接近论文第一阶段。

### 2.2 `stage2`

目标是贴近论文第二阶段。

当前实现包含：

- `vision_model`
- `Q-Former`
- `language_projection`
- `LLM`

实现方式是：

- 直接使用 Hugging Face 的 `Blip2ForConditionalGeneration`
- 默认冻结 LLM
- 训练 `Q-Former` 和桥接投影层
- 把图像信息作为视觉前缀注入 LLM

## 3. 这和原论文还有哪些差距

虽然现在已经比之前更接近原论文，但它还不是原论文训练代码的逐行复刻。

主要差距有：

- 没有复现论文的大规模预训练数据与训练时长
- `stage1` 里的 `ITG` 是基于当前 Hugging Face `Q-Former` 组件补出来的仓库内实现
- `ITM` 的负样本来自 batch 内 hard negative mining
  - 不是完整复现原始大规模数据采样策略
- `stage2` 使用的是 Hugging Face 的 `Blip2ForConditionalGeneration`
  - 训练方式更工程化，便于在当前仓库里落地

所以更准确的说法是：

- 这版实现是“更接近论文 BLIP-2 的工程版训练链路”
- 不是“官方 BLIP-2 预训练代码原样移植”

## 4. 环境准备

最少依赖：

```bash
pip install torch torchvision transformers pillow
```

如果你用 GPU，`torch` 需要安装和机器 CUDA 匹配的版本。

环境验证：

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python3 -c "from transformers import Blip2ForImageTextRetrieval, Blip2ForConditionalGeneration; print('ok')"
```

如果第二条命令报错，说明当前环境里的 `transformers` 版本不支持这两条 BLIP-2 链路。

## 5. 数据准备

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

`blip2/` 没有自己再写一套 dataset，而是直接复用 [clip/data.py](/home/by/workspace/MLLM/clip/data.py:1)。

它会把原始记录转成 `CLIPSample`：

- `sample_id`
- `image_path`
- `text`
- `record`

其中：

- `text`
  - 主要服务 `stage1`
  - 按 `text_mode` 构造
- `record`
  - 在 `stage2` 里会继续拿来构造 prompt 和 target

## 6. 文本构造方式

因为 `blip2/` 直接复用 `clip/data.py`，所以当前也支持 4 种 `text_mode`：

- `assistant`
  - 只使用答案文本
- `user`
  - 只使用问题文本
- `qa`
  - 问题和答案直接拼接
- `assistant_with_question`
  - 格式化为 `Question: ... / Answer: ...`

建议：

- `stage1`
  - 先用 `assistant`
  - 更适合图文对齐
- `stage2`
  - `text_mode` 仍然会影响 dataset 里 `sample.text`
  - 但真正的生成式 prompt / target 主要来自：
    - `user -> prompt`
    - `assistant -> target`

## 7. 配置文件怎么改

### 7.1 `stage1` 默认配置

默认配置文件是 [blip2/config.json](/home/by/workspace/MLLM/blip2/config.json:1)。

关键字段：

- `training_stage`
  - 固定为 `stage1`
- `model_name`
  - 默认是 `Salesforce/blip2-itm-vit-g`
- `itc_weight`
  - `ITC` 损失权重
- `itm_weight`
  - `ITM` 损失权重
- `itg_weight`
  - `ITG` 损失权重
- `freeze_vision`
  - 默认 `true`
- `freeze_qformer`
  - 控制是否冻结 `Q-Former`
- `freeze_text_embeddings`
  - 控制是否冻结文本 embedding
- `freeze_projection`
  - 控制是否冻结检索投影层和 `itm_head`

### 7.2 `stage2` 示例配置

示例配置文件是 [blip2/stage2_config.json](/home/by/workspace/MLLM/blip2/stage2_config.json:1)。

关键字段：

- `training_stage`
  - 固定为 `stage2`
- `model_name`
  - 示例里是 `Salesforce/blip2-flan-t5-xl`
- `freeze_language_model`
  - 默认 `true`
  - 更接近论文“冻结 LLM”的做法
- `freeze_language_projection`
  - 是否冻结从 `Q-Former` 到 LLM embedding 空间的投影层
- `generation_prompt_template`
  - 默认是：
  - `Question: {user}\nAnswer:`
- `target_max_length`
  - stage2 target 文本的最大长度

## 8. 怎么跑

### 8.1 运行 `stage1`

```bash
python3 main.py --config blip2/config.json
```

### 8.2 运行 `stage2`

```bash
python3 main.py --config blip2/stage2_config.json
```

如果你要直接跑模型目录下脚本，也可以：

```bash
python3 blip2/main.py --config blip2/config.json
python3 blip2/main.py --config blip2/stage2_config.json
```

## 9. 数据流变化总览

### 9.1 `stage1` 数据流

```text
JSONL
  -> CLIPJsonlDataset
  -> CLIPBatchCollator
  -> AutoProcessor(Blip2Processor)
  -> pixel_values / input_ids / attention_mask
  -> BLIP2Stage1Model.forward(...)
  -> ITC + ITM + ITG
```

### 9.2 `stage2` 数据流

```text
JSONL
  -> CLIPJsonlDataset
  -> BLIP2Stage2BatchCollator
  -> prompt from user
  -> target from assistant
  -> AutoProcessor(Blip2Processor)
  -> pixel_values / input_ids / labels
  -> BLIP2Stage2Model.forward(...)
  -> generative LM loss
```

## 10. `stage1` 的具体模型与张量流

### 10.1 图像分支

调用链：

```text
pixel_values
  -> vision_model
  -> image_hidden_states
  -> query_tokens
  -> qformer(cross-attention to image)
  -> vision_projection
  -> normalize
```

默认 checkpoint 下，主要 shape 可以按下面理解：

- `pixel_values`
  - `[B, 3, 224, 224]`
- `image_embeds`
  - `[B, 32, 256]`

这里和 `CLIP` 最大的不同是：

- `CLIP` 一张图只有一个全局向量
- 这里一张图先产生 `32` 个 query-level 向量

### 10.2 文本分支

调用链：

```text
input_ids / attention_mask
  -> embeddings
  -> qformer(query_length=0)
  -> text_projection
  -> normalize
```

主要 shape：

- `input_ids`
  - `[B, L]`
- `text_embeds`
  - `[B, 256]`

### 10.3 `ITC`

相似度计算：

```text
image_embeds [B, 32, 256]
  @ text_embeds.t() [256, B]
  -> [B, 32, B]
  -> max over query axis
  -> logits_per_image [B, B]
  -> logits_per_text [B, B]
```

然后做双向对比学习：

- `image -> text`
- `text -> image`

### 10.4 `ITM`

当前仓库实现会做 batch 内 hard negative mining：

- 正样本：
  - `(image_i, text_i)`
- 文本负样本：
  - `(image_i, text_j_hard)`
- 图像负样本：
  - `(image_k_hard, text_i)`

然后通过跨模态 `Q-Former` + `itm_head` 输出二分类 logits：

- `itm_logits`
  - shape: `[3B, 2]`

### 10.5 `ITG`

这一步是这次改造里新增的重点。

调用链：

```text
query_tokens + text embeddings
  -> custom stage1 attention mask
  -> qformer(query_length=32, cross-attend image)
  -> text hidden states
  -> itg_head
  -> token logits
```

这里的 attention mask 约束是：

- query token 只看 query token
- text token 可以看所有 query token
- text token 对 text token 使用 causal mask

这就是当前实现里“更接近论文第一阶段 image-grounded text generation”的部分。

## 11. `stage2` 的具体模型与张量流

`stage2` 走的是 `Blip2ForConditionalGeneration`。

调用链：

```text
pixel_values
  -> vision_model
  -> qformer(query tokens)
  -> language_projection
  -> visual prefix / soft prompt
  -> LLM
  -> generative loss
```

### 11.1 prompt / target 怎么构造

当前 `stage2` collator 会：

- 从 `user` 字段清掉 `<image>`
- 用 `generation_prompt_template` 生成 prompt
- 默认模板：
  - `Question: {user}\nAnswer:`
- target 默认取：
  - `assistant`
  - 如果没有 `assistant`，退回 `sample.text`

### 11.2 encoder-decoder 和 decoder-only

当前实现同时兼容两类 LLM：

- encoder-decoder
  - 例如 `FlanT5`
  - prompt 作为输入
  - target 作为 labels
- decoder-only
  - 例如 `OPT`
  - prompt + target 拼成同一串文本
  - labels 会自动把：
    - prompt 部分
    - padding
    - `<image>` 占位 token
    置成 `-100`

## 12. 日志和指标

### 12.1 `stage1`

训练日志会看这些：

- `loss`
- `image_acc`
- `text_acc`
- `itc_loss`
- `itm_loss`
- `itg_loss`

### 12.2 `stage2`

训练日志会看这些：

- `loss`
- `token_acc`

这里的 `token_acc` 是 teacher-forcing 下、忽略 `-100` label 后的 token 级准确率。

## 13. checkpoint 保存了什么

保存后通常是：

```text
outputs/.../
└── checkpoints/
    └── best/
        ├── model/
        ├── processor/
        └── training_state.pt
```

其中：

- `processor/`
  - 保存当前 `AutoProcessor`
- `training_state.pt`
  - 保存 epoch、metrics、training_stage

`model/` 在两个阶段略有区别：

- `stage1`
  - 会保存：
    - `backbone/`
    - `stage1_state.pt`
    - `stage1_metadata.json`
- `stage2`
  - 直接保存 Hugging Face `Blip2ForConditionalGeneration`

## 14. 一句话总结

如果只用一句话概括当前 `blip2/`：

- `stage1` 现在已经从“只做 retrieval”升级成“更接近论文第一阶段的 `ITC + ITM + ITG`”
- `stage2` 则提供了把 `Q-Former` 接到 frozen LLM 的生成式训练链路
