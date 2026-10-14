# 快速开始

[返回首页](../README.md) · [使用指南](usage.md) · [接口参考](reference.md)

本页提供无需 STEP 文件的最小训练与推理示例。示例验证调用流程；处理真实 CAD 数据时，应继续阅读数据准备和验证范围。

## 环境要求

| 依赖 | 作用 |
| --- | --- |
| Python 3.10+ | 核心运行环境 |
| NumPy | 数值训练、批量推理与检索 |
| CadQuery/OpenCascade | 可选；STEP 读取及几何特征提取 |
| pytest | 可选；运行仓库测试 |

核心无需 GPU、AI 账号或预训练权重。几何 VAE 与结构 VAE 是分别训练的模型，不共享一套潜空间权重。

## 安装

```bash
git clone https://github.com/kkppshuai-svg/formalatent.git
cd formalatent
```

Linux/Bash：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

PowerShell 示例后续命令中的 `python` 应指向该环境。可以激活虚拟环境，也可以将每个 `python` 替换为 `& ./.venv/Scripts/python.exe`。

## 生成示例数据

```bash
python examples/create_demo_dataset.py --out-dir data/demo
```

脚本使用固定种子 42，生成 32 条三维合成记录和两个查询。文件为 UTF-8 JSON：

| 文件 | 内容 |
| --- | --- |
| `data/demo/geometry.jsonl` | 每行一个样本，特征名为 `demo.x`、`demo.y`、`demo.z` |
| `data/demo/query.json` | 形状为 `(2, 3)` 的查询数组 |

这些向量不是从 STEP 提取的特征。查询来自演示样本表，因此检索结果不能衡量未见零件的泛化能力。

## 训练与推理

```bash
python scripts/train_brep_vae.py --dataset data/demo/geometry.jsonl --out models/demo.json --latent-dim 2 --hidden-dim 16 --epochs 100 --seed 42
python scripts/vae_cli.py encode --model models/demo.json --input data/demo/query.json --out data/demo/latents.json
python scripts/vae_cli.py decode --model models/demo.json --input data/demo/latents.json --out data/demo/reconstructed.json
python scripts/vae_cli.py search --model models/demo.json --input data/demo/query.json --limit 3
```

训练器创建模型输出目录；数据准备脚本创建演示数据目录。预期产物：

| 文件 | 含义 |
| --- | --- |
| `models/demo.json` | NumPy 模型参数、归一化统计、样本索引与训练元数据 |
| `data/demo/latents.json` | `(2, 2)` 潜向量数组 |
| `data/demo/reconstructed.json` | `(2, 3)` 描述重建数组 |
| 检索标准输出 | 每个查询最多三个匹配项，包含 ID、距离及分数 |

较小的混合 `score` 表示更近，不是匹配概率。示例运行摘要见[演示记录](../examples/demo-result.json)。随机种子固定，但不同数值库或平台可能有浮点差异。

## 处理 STEP 文件

```bash
python -m pip install -r requirements-cad.txt
python scripts/extract_brep_geometry.py --step path/to/part.step
```

需要准备多样本清单，批量提取几何 JSONL，并使用内置 44 维 schema 训练模型，之后才能执行 STEP 查询。完整流程见[使用指南](usage.md)。三维合成模型不兼容 STEP 查询。

## 验证安装

```bash
python -m pip install -r requirements-dev.txt
```

Bash：

```bash
PYTHONPATH=scripts python -m pytest scripts -q
```

PowerShell：

```powershell
$env:PYTHONPATH = "scripts"
python -m pytest scripts -q
```

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 找不到 NumPy 或 CadQuery | 核对依赖安装到当前执行命令所用的 Python 环境 |
| 特征 schema 不一致 | 保持名称、顺序和维数完全一致；不要混用合成与 STEP 特征 |
| `vae_runtime` 导入失败 | Python 集成时设置 `PYTHONPATH=scripts` |
| `decode` 没有生成 STEP | 当前解码输出是统计描述，实体生成不在本项目范围内 |
| 不能复现历史 505 组指标 | 仓库不分发原始数据或权重，需要按[训练记录](deepcad-training.md)准备相同输入与划分 |
