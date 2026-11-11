# FormaLatent documentation / 文档中心

[English overview](../README.md) · [中文首页](../README.zh-CN.md)

| Topic / 主题 | English | 中文 |
| --- | --- | --- |
| Getting started / 快速开始 | [Guide](getting-started.en.md) | [指南](getting-started.md) |
| Technology stack / 技术栈 | [Stack](tech-stack.en.md) | [技术栈](tech-stack.md) |
| Neural matrices / 神经矩阵 | [Architecture](neural-network.en.md) | [网络与矩阵](neural-network.md) |
| Data and training / 数据与训练 | — | [使用指南](usage.md) |
| CLI and Python API / 接口 | — | [接口参考](reference.md) |
| Structure VAE / 结构模型 | — | [结构 VAE](structure-vae.md) |
| AI-CAD joint completion / 关节补全 | [Getting started](getting-started.en.md#ai-cad-joint-completion) | [结构图训练](aicad-graph-training.md) · [运行与运动验证](mechanism-feedback.md) |
| Validation / 验证 | — | [验证范围](validation.md) |
| Recorded experiments / 实验记录 | — | [DeepCAD 505 组](deepcad-training.md) · [多种子评估](evaluation.md) |
| Contributing / 贡献 | — | [贡献规范](../CONTRIBUTING.md) |

## Project statements

FormaLatent was extracted from [AI-CAD](https://github.com/kkppshuai-svg/ai-cad), with geometric implementation baseline `52f44be`. It provides learning and retrieval. The joint completion command uses the AI-CAD builder and CadQuery to verify candidates at sampled positions.

Historical experiments retain metrics and provenance, but raw training data and pretrained weights are not distributed. Generated models may contain sample IDs, names, and descriptors. No standalone license text is currently included; third-party code and data remain subject to their respective authorization conditions.

FormaLatent 源自 AI-CAD，几何实现基线为 `52f44be`，提供学习、检索和关节候选排序。关节补全使用 AI-CAD 构建器与 CadQuery 检查采样位置。仓库保留实验指标和来源记录，不分发原始数据或预训练权重。生成模型可能包含样本 ID、名称及描述。仓库尚未附独立许可证文本，第三方代码和数据按各自授权条件使用。
