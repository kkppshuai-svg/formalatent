# FormaLatent · 形潜

**面向 CAD 几何的潜空间学习与检索。**

*Part of AI-CAD*

FormaLatent 从 STEP 模型提取几何统计特征，提供模型训练、批量编码、描述向量重建及相似样本检索。项目采用 Python 与 NumPy 实现，可独立于 AI-CAD Web 服务运行；STEP 特征提取依赖 CadQuery。

| 项目 | 说明 |
| --- | --- |
| 几何训练实现 | 3.0.0（模型训练元数据中的版本标识） |
| 结构神经模型 | `formalatent-structure-vae-v1` |
| 模型格式 | `ai-cad-brep-vae-v2` |
| 几何特征 | 内置 STEP 提取器输出 44 维描述向量 |
| 运行环境 | Python 3.10+；依赖见 `requirements*.txt` |
| 上游来源 | [kkppshuai-svg/ai-cad](https://github.com/kkppshuai-svg/ai-cad)，基线提交 `52f44be` |

## 能力范围

- **几何表征**：提取拓扑数量、包围盒比例、质量属性、曲面与曲线类型及邻接统计。
- **模型训练**：支持小批量 Adam、KL 权重预热、验证集早停与预训练模型微调。
- **批量推理**：缓存模型权重，提供编码、后验参数查询、解码与重建接口。
- **相似检索**：结合潜空间距离与几何描述距离；旧模型可回退到潜空间检索。

解码结果是几何统计描述向量，**不包含生成 STEP 实体所需的完整拓扑和参数信息**。仓库还提供[文本结构神经 VAE](docs/structure-vae.md)，并保留非神经网络的 SVD 基线作为对照。训练器支持自定义维数；使用 STEP 查询时，特征名称及顺序必须与内置提取器一致。

## 快速开始

以下命令在仓库根目录执行，适用于 Bash 环境。

### 1. 安装

```bash
git clone https://github.com/kkppshuai-svg/formalatent.git
cd formalatent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

如需处理 STEP 文件，再安装几何依赖：

```bash
python -m pip install -r requirements-cad.txt
```

### 2. 运行最小示例

以下示例创建三维合成数据，用于验证训练与推理流程；不用于衡量 CAD 建模效果。

```bash
mkdir -p data models
python - <<'PY'
import json
from pathlib import Path
import numpy as np

vectors = np.random.default_rng(42).normal(size=(32, 3))
records = [
    {"id": f"demo-{i}", "geometry": {
        "featureNames": ["demo.x", "demo.y", "demo.z"],
        "vector": vector.tolist()
    }}
    for i, vector in enumerate(vectors)
]
Path("data/demo.jsonl").write_text(
    "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
)
Path("data/query.json").write_text(json.dumps(vectors[:2].tolist()), encoding="utf-8")
PY
python scripts/train_brep_vae.py --dataset data/demo.jsonl --out models/demo.json
python scripts/vae_cli.py encode --model models/demo.json --input data/query.json --out data/latents.json
python scripts/vae_cli.py decode --model models/demo.json --input data/latents.json --out data/reconstructed.json
python scripts/vae_cli.py search --model models/demo.json --input data/query.json --limit 3
```

`encode` 和 `decode` 输出二维数值数组；`search` 为每个输入向量返回一个匹配列表。真实 STEP 数据的准备方式见[使用指南](docs/usage.md)。

## 最新训练快照

2026-09-23 使用 AnySearch 比较 DeepCAD、Fusion 360 Gallery、SketchGraphs 和 ABC Dataset 后，选择带官方构造序列与训练/测试划分的 DeepCAD。530 个候选经过实体有效性、STEP 往返和 44 维描述去重后保留 **505 组**：415 组用于训练域，90 组官方 test 全程隔离。

| 指标 | 505 组运行 | 上一版 301 组运行 | 变化 |
| --- | ---: | ---: | ---: |
| 结构标签 Brier 误差 ↓ | 0.004058 | 0.006691 | 降低 39.35% |
| 数量标准化 log1p MSE ↓ | 0.020149 | 0.111549 | 降低 81.94% |
| 标签 Jaccard 最邻近一致率 ↑ | 92.22% | 94.92% | 降低 2.69 个百分点 |
| 几何标准化重建 MSE ↓ | 0.057888 | 0.142034 | 降低 59.24% |

结构 VAE 的标签 Brier 误差优于同输入 SVD 对照 0.009994，检索代理一致率也高于 SVD 的 91.11%；数量重建仍明显落后于 SVD，因此不宣称全面领先。505 个样本全部完成编码与解码，独立重载审计通过。训练配置、数据源取舍、对照指标和复现命令见[最新外部样本训练记录](docs/deepcad-training.md)。

## 文档

| 文档 | 内容 |
| --- | --- |
| [使用指南](docs/usage.md) | 数据准备、训练、微调、STEP 查询与常见问题 |
| [接口与数据格式](docs/reference.md) | CLI 参数、Python API、模型字段与兼容性 |
| [评估报告](docs/evaluation.md) | 实验方法、原始结果、指标解释与复现条件 |
| [文本结构神经 VAE](docs/structure-vae.md) | 标签与数量的神经编码、解码、SVD 对照及 AI-CAD 接入 |
| [外部样本训练记录](docs/deepcad-training.md) | AnySearch 来源、样本筛选、独立测试与实际训练产物 |
| [变更记录](CHANGELOG.md) | 3.0.0 的功能与行为变更 |

## 验证

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH=scripts python -m pytest scripts -q
```

测试涵盖特征提取、训练、微调、数据划分、批量运行时、CLI 及文本结构基线。小规模实验中，100 个 DeepCAD 样本在三个固定种子下的验证集原始描述 MSE 下降约 13%–29%；12 个本地样本上的结果有升有降。完整数值、时间开销与适用范围见[评估报告](docs/evaluation.md)。

## 仓库结构

```text
scripts/          训练、推理、特征提取及测试
benchmarks/       对比脚本与已记录的实验结果
docs/             使用指南、接口参考与评估说明
requirements*.txt 分层依赖清单
CHANGELOG.md      变更记录
```

仓库不分发原始训练数据或预训练权重。生成的模型包含样本 ID、名称及几何描述向量，分享模型前应核对其中的数据。本仓库当前未提供独立许可证文件；使用及再分发时应确认相关代码和数据的授权条件。
