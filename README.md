# MLLM

这个仓库用于管理多模态训练代码与数据整理脚本，当前先落地了一套 `CLIP` 训练链路。

## 当前结构

```text
MLLM/
├── main.py
├── clip/
│   ├── config.json
│   ├── data.py
│   ├── main.py
│   ├── model.py
│   └── README.md
├── scripts/
│   ├── prepare_ureader_kg.py
│   └── organize_existing_subset.py
├── data/
├── datasets/
├── artifacts/
└── outputs/
```

## 当前约定

- 每个模型单独放在自己的目录下，例如 `clip/`
- 模型自己的配置也放在对应目录里，不放全局配置文件
- 仓库根目录只保留统一入口 `main.py`

## CLIP 的使用方式

只需要两步：

1. 修改 [clip/config.json](/home/by/workspace/MLLM/clip/config.json:1)
2. 在仓库根目录运行：

```bash
python3 main.py
```

`clip/config.json` 里主要改这些字段：

- `train_annotations`
- `dataset_root`
- `output_dir`
- `model_name`
- `text_mode`
- `epochs`
- `batch_size`
- `eval_batch_size`
- `num_workers`
- `val_ratio`

说明：

- 路径统一按仓库根目录解析
- 现在默认就是全量训练，不再保留抽样 demo 流程
- `main.py` 会读取 `clip/config.json`，再调用 `clip/` 下的训练逻辑

## 数据整理脚本

- `scripts/prepare_ureader_kg.py`
  - 修复原始 `ureader_kg_processed.json`
  - 导出清洗后的 `JSONL`
  - 可选过滤缺失图片的样本
- `scripts/organize_existing_subset.py`
  - 把当前可用图片整理成训练可直接读取的数据集目录
  - 重写图片路径到本地相对路径

## 说明

- 原始图片、模型权重和训练输出不提交到 git
- 当前默认训练集路径是 `datasets/ureader_existing_local/annotations/train.jsonl`
- 如果后续新增别的模型目录，沿用同样的组织方式即可
