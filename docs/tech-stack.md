# 技术栈与模块关系

[中文首页](../README.zh-CN.md) · [English](tech-stack.en.md) · [神经网络矩阵](neural-network.md)

## 分层依赖

| 层次 | 技术及约束 | 依赖来源 | 承担工作 |
| --- | --- | --- | --- |
| 独立核心 | Python 3.10+、NumPy ≥1.24,<3 | [requirements.txt](../requirements.txt) | 数值训练、矩阵推理、编码、解码与检索 |
| 几何提取 | CadQuery ≥2.4,<3，底层 OpenCascade | [requirements-cad.txt](../requirements-cad.txt) | STEP/BRep 读取、拓扑与几何统计 |
| 开发测试 | pytest ≥7,<10，加上核心与 CAD 依赖 | [requirements-dev.txt](../requirements-dev.txt) | 训练、数据、几何和运行时回归检查 |
| 上游工作台集成 | JavaScript ES modules、Node.js | [AI-CAD 集成目录](../integrations/ai-cad) | 结构 VAE 后验均值推理、模型加载核对和接入补丁 |

核心训练和数值查询只依赖 Python 与 NumPy；读取 STEP 时再安装 CAD 依赖。OpenCascade 由 CadQuery 的几何绑定提供，仓库未在依赖清单中单独固定 OCCT 版本。`requirements-dev.txt` 包含 CAD 依赖，因此完整测试环境比纯数值运行环境更完整。

JavaScript 模块用于 AI-CAD 接入，不是独立 Python 核心的必需运行时。它实现结构标签的确定性后验均值计算；几何训练与批量运行时仍由 Python 实现。

## 神经网络实现

几何及结构 VAE 使用 NumPy `ndarray` 与 `@` 矩阵乘法，包含五个全连接权重矩阵、五个偏置向量、tanh 激活和高斯后验分支。训练时执行重参数采样、显式链式梯度和 Adam 更新。

这是用 NumPy 实现的可训练神经网络，核心依赖清单没有额外的深度学习框架。SVD 是单独保留的非神经基线。矩阵形状、参数数量、损失公式和实际权重查看方法见[神经网络与矩阵计算](neural-network.md)。

## 源码职责

| 入口 | 职责 |
| --- | --- |
| [train_brep_vae.py](../scripts/train_brep_vae.py) | 几何 VAE 参数初始化、前向、梯度、Adam、训练与模型保存 |
| [structure_vae.py](../scripts/structure_vae.py) | 结构标签/数量向量化、结构目标、训练及运行时 |
| [vae_runtime.py](../scripts/vae_runtime.py) | 校验并缓存几何模型数组，执行批量推理与检索 |
| [vae_cli.py](../scripts/vae_cli.py) | argparse CLI 与 JSON 输入输出 |
| [extract_brep_geometry.py](../scripts/extract_brep_geometry.py) | STEP 几何特征与 JSONL 数据集提取 |
| [prepare_deepcad_training.py](../scripts/prepare_deepcad_training.py) | 已支持 DeepCAD 序列的重建、样本筛选及来源记录 |
| [run_training_experiment.py](../scripts/run_training_experiment.py) | 双模型实验、NPZ 产物及独立测试指标 |
| [structure-vae-runtime.mjs](../integrations/ai-cad/structure-vae-runtime.mjs) | JavaScript 结构后验均值推理 |
| [verify-model.mjs](../integrations/ai-cad/verify-model.mjs) | AI-CAD 接入的模型核对入口 |

## 数据与模型

训练样本使用 UTF-8 JSONL，几何模型及结构模型参数使用 JSON。实验批量编码和重建可保存为压缩 NPZ。`weights` 包含可训练矩阵及偏置；归一化统计、词表、样本索引和指标分别保存。

DeepCAD 是已有实验的数据来源，不是所有运行场景必需的组件。数据、模型格式与依赖版本有不同生命周期，不能仅凭同一模型格式认定两个实验可直接比较。

## 与其他项目的关系

FormaLatent 可独立运行，同时保留 AI-CAD 集成模块。STEP 提取使用 CadQuery/OpenCascade，当前没有直接的 NVIDIA USD BRep 特征适配器。[USD BRep Review](https://github.com/kkppshuai-svg/usd-brep-review) 是独立审查层，不属于本项目的核心依赖。

表示学习、确定性建模和实际几何审查分别承担不同职责。将它们接入同一个工作台时，学习或检索结果仍需经过实际建模与审查流程。
