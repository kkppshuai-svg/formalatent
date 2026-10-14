# FormaLatent · 形潜

**面向 CAD 几何与结构的神经表征和相似检索。**

[English](README.md) · [快速开始](docs/getting-started.md) · [文档中心](docs/README.md)

使用 NumPy VAE 编码 CAD 描述、重建特征并检索相似样本，可独立于 AI-CAD 工作台运行。

![几何与结构双通道](docs/assets/formalatent-overview.svg)

## 功能

- 几何与结构神经 VAE：可学习矩阵、Adam 和批量推理。
- CadQuery/OpenCascade 提取 STEP 描述，支持潜空间与描述混合检索。
- Python 和 CLI 提供训练、微调、编码、解码及查询。

解码输出是描述，不是完整 STEP 实体。仓库不分发原始数据或预训练权重。

**技术栈：** Python · NumPy · CadQuery/OpenCascade（STEP 可选依赖）。[详细说明](docs/tech-stack.md)

## 快速开始

Python 3.10+，合成示例只需 NumPy：

```bash
git clone https://github.com/kkppshuai-svg/formalatent.git
cd formalatent
python -m pip install -r requirements.txt
python examples/create_demo_dataset.py
python scripts/train_brep_vae.py --dataset data/demo/geometry.jsonl --out models/demo.json --latent-dim 2 --hidden-dim 16 --epochs 100
python scripts/vae_cli.py search --model models/demo.json --input data/demo/query.json --limit 3
```

环境配置、STEP 输入和编码/解码见[快速开始](docs/getting-started.md)。

## 文档

- [CLI 与 Python API](docs/reference.md)
- [神经网络与矩阵](docs/neural-network.md)
- [DeepCAD 实验记录](docs/deepcad-training.md) · [验证边界](docs/validation.md)
- [贡献规范](CONTRIBUTING.md) · [问题反馈](https://github.com/kkppshuai-svg/formalatent/issues)

源自 [AI-CAD](https://github.com/kkppshuai-svg/ai-cad)。仓库尚未附独立许可证文本，见[项目声明](docs/README.md#project-statements)。
