# 接口与数据格式

[返回首页](../README.md) · [使用指南](usage.md)

## 训练命令

入口：`python scripts/train_brep_vae.py`。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--dataset` | `cad-latent/brep_training_samples.jsonl` | 几何 JSONL 输入 |
| `--out` | `cad-latent/brep_vae_model.json` | 模型 JSON 输出 |
| `--latent-dim` | `8` | 潜空间维数 |
| `--hidden-dim` | `64` | 编码器与解码器隐藏层宽度 |
| `--epochs` | `400` | 最大训练轮数 |
| `--beta` | `0.01` | 目标 KL 权重 |
| `--learning-rate` | `0.01` | Adam 学习率 |
| `--seed` | `42` | 随机种子 |
| `--batch-size` | `64` | 小批量大小 |
| `--warmup-epochs` | `20` | KL 预热轮数；`0` 表示无预热 |
| `--init-model` | 无 | 微调起始模型 |

默认数据路径沿用 AI-CAD，独立仓库不附带这些文件。建议显式传入数据和输出路径。微调时维数取自预训练模型。

## 批量 CLI

```text
python scripts/vae_cli.py {encode,decode,reconstruct,search}
    --model MODEL.json --input INPUT.json
    [--out OUTPUT.json] [--limit 5] [--geometry-weight 0.5]
```

`--model` 和 `--input` 必填。未指定 `--out` 时输出 JSON 到标准输出。`encode`、`reconstruct` 与 `search` 接收形状为 `(N, inputDim)` 的数组；`decode` 接收 `(N, latentDim)` 的数组。检索参数仅用于 `search`。

## Python API

从仓库根目录设置 `PYTHONPATH=scripts`，再运行调用代码：

```python
import json
from pathlib import Path
from train_brep_vae import load_model
from vae_runtime import VaeRuntime

runtime = VaeRuntime(load_model("models/vae.json"))
vectors = json.loads(Path("data/query.json").read_text(encoding="utf-8"))
latent = runtime.encode(vectors)
mu, logvar = runtime.encode(vectors, posterior=True)
reconstructed = runtime.decode(latent)
matches = runtime.search(vectors, limit=5, geometry_weight=0.5)
```

| 接口 | 返回值 |
| --- | --- |
| `VaeRuntime(model)` | 校验并缓存 NumPy 权重、归一化参数及样本索引 |
| `encode(vectors)` | 后验均值，形状 `(N, latentDim)` |
| `encode(vectors, posterior=True)` | 均值及对数方差数组；对数方差裁剪到 `[-8, 5]` |
| `decode(latents)` | 原始特征尺度下的描述，形状 `(N, inputDim)` |
| `reconstruct(vectors)` | 等价于 `decode(encode(vectors))` |
| `search(vectors, limit=5, geometry_weight=0.5)` | 每个查询对应一个匹配列表，按得分升序排列 |

推理使用确定性的后验均值；返回后验参数不会触发随机采样。解码器输出没有逐特征物理约束，数量或比例类字段可能超出有效范围。

### 检索得分

设潜空间维数为 `L`、输入维数为 `D`、几何权重为 `w`：

```text
latent_distance = ||z_query - z_sample||₂
geometry_distance = ||standardized_query - standardized_sample||₂ / sqrt(D)
score = (1 - w) * latent_distance / sqrt(L) + w * geometry_distance
```

全部索引样本均具有 `geometryVector` 时，才启用几何项；否则整体回退到 `latent_distance / sqrt(L)`。相同得分按样本 ID 的字符串顺序排列。

每条匹配包含 `id`、`name`、`assemblyId`、`distance`、`score` 和 `geometryDistance`。其中 `distance` 是未按维数缩放的潜空间距离，**排序依据是 `score`**；几何项未参与时 `geometryDistance` 为 `null`。检索是精确扫描，没有近似最近邻索引。

## 模型文件

| 字段 | 内容 |
| --- | --- |
| `format` | `ai-cad-brep-vae-v2` |
| `sampleCount`、`inputDim`、`hiddenDim`、`latentDim` | 样本数与网络维数 |
| `featureNames` | 输入特征的有序名称列表 |
| `normalization.mean`、`normalization.std` | 逐特征归一化统计 |
| `weights` | 编码器、均值/方差分支与解码器参数 |
| `training` | 超参数、已执行轮数、数据划分索引及实现版本 |
| `metrics.reconstructionMse` | 最终权重在全部输入记录上的归一化重建 MSE |
| `metrics.klLoss` | 最终权重在全部输入记录上的平均 KL 项 |
| `metrics.validationLoss` | 选择权重时取得的最优验证目标值 |
| `samples` | 样本元数据、潜向量及可选几何向量 |

`trainIndices` 与 `validationIndices` 指向加载后的有效记录顺序，不是原始 JSONL 行号。`epochsCompleted` 是执行轮数，不是最优权重的轮数。模型不保存优化器状态或完整训练历史。

## 兼容性

3.0.0 是实现版本，模型格式仍为 v2，二者独立。运行时支持具有合法维数与权重的旧 v2 模型；没有几何向量的模型使用潜空间检索。新增字段不会改变既有权重布局。

文本结构基线入口为 `train_cad_vae.py` 和 `query_cad_vae.py`，使用独立模型格式，不可传入 `VaeRuntime`。CLI 的完整参数可通过各脚本的 `--help` 查看。
