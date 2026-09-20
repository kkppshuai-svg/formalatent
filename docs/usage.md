# 使用指南

[返回首页](../README.md) · [接口参考](reference.md)

## 数据准备

训练输入采用 UTF-8 JSONL，每行一个样本。最小记录如下：

```json
{"id":"part-001","name":"示例零件","geometry":{"featureNames":["feature.a","feature.b"],"vector":[0.2,0.8]}}
```

所有有效记录必须具有相同的特征名称、顺序及向量长度；数值必须有限，至少提供两个有效样本。建议使用唯一的 `id`，并在训练前去除重复或近重复零件。缺失几何特征的记录会被加载器跳过；特征顺序不一致或数值不合法会导致训练失败。

### 从 STEP 构建训练集

安装 `requirements-cad.txt` 后，可先检查单个文件：

```bash
python scripts/extract_brep_geometry.py --step /path/to/part.step
```

准备样本清单 `data/parts.jsonl`，每条记录提供 `id` 和 `stepPath`：

```json
{"id":"part-001","name":"零件一","stepPath":"/path/to/part-001.step"}
{"id":"part-002","name":"零件二","stepPath":"/path/to/part-002.step"}
```

批量提取：

```bash
python scripts/extract_brep_geometry.py --dataset data/parts.jsonl --out data/geometry.jsonl
```

也可提供 `assemblyId`，通过 `--assemblies /path/to/assemblies` 查找 `<assemblyId>/assembly.step`。相对路径以命令执行目录为基准。提取器会跳过无法处理的样本，并在标准输出报告 `sampleCount`、`failureCount` 和失败详情；训练前应检查这些字段。

## 训练

```bash
python scripts/train_brep_vae.py \
  --dataset data/geometry.jsonl \
  --out models/vae.json \
  --latent-dim 8 --hidden-dim 64 \
  --epochs 400 --batch-size 64 --warmup-epochs 20 --seed 42
```

训练器按固定种子打乱样本，将约 20% 留作验证；少于五条记录时保留一条用于验证。归一化统计仅使用训练集，标准差小于 `1e-8` 的特征按标准差 `1` 处理。

优化目标由归一化重建 MSE 与 KL 项组成。KL 权重在预热期间线性增加；验证时始终使用完整的目标 KL 权重。连续 60 个 epoch 未达到 `1e-7` 的验证目标改善时停止，最终返回验证目标最优的权重，初始权重也参与候选比较。

划分基于记录而非零件家族；相关零件可能进入不同子集。正式评估应另行准备独立测试集，避免仅凭训练器的验证指标判断泛化能力。

## 微调

```bash
python scripts/train_brep_vae.py \
  --dataset data/finetune.jsonl \
  --init-model models/vae.json \
  --out models/tuned.json \
  --learning-rate 0.003
```

微调继承完整权重、隐藏维数、潜空间维数及归一化统计。目标数据的特征名称和顺序必须匹配。Adam 状态重新初始化，因此该功能是微调，不是对中断训练的精确续跑。若目标域差异较大，建议比较微调与从头训练的独立测试表现。

## 推理与检索

STEP 查询要求模型使用内置提取器的 44 维特征：

```bash
python scripts/query_brep_vae.py --model models/vae.json --step /path/to/query.step --limit 5
```

批量描述查询与重建：

```bash
python scripts/vae_cli.py search --model models/vae.json --input data/query.json --limit 5 --geometry-weight 0.5
python scripts/vae_cli.py reconstruct --model models/vae.json --input data/query.json --out data/reconstructed.json
```

输入文件是二维数组，即使只有一个查询也要保留外层数组。输出路径的父目录应预先存在。`--geometry-weight 0` 使用纯潜空间检索；值为 `1` 时，在几何索引可用的情况下使用纯描述距离。

## 常见问题

| 现象 | 检查方法 |
| --- | --- |
| `No module named cadquery` | 在当前 Python 环境安装 `requirements-cad.txt`；仅处理数值向量无需此依赖 |
| `No module named vae_runtime` | Python 集成时设置 `PYTHONPATH=scripts`，并从仓库根目录执行 |
| 特征 schema 不匹配 | 核对 `featureNames` 的名称、顺序和维数；合成示例模型不能用于 STEP 查询 |
| 样本不足 | 检查提取失败报告及训练记录是否包含有效 `geometry` |
| 解码结果不是实体 | 解码输出是描述向量；当前项目没有向量到完整 BREP 的实体生成器 |
| 微调后效果下降 | 检查域差异、样本重复、学习率及独立测试集；旧归一化统计会被保留 |
