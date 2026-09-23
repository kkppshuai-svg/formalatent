# DeepCAD 505 组外部样本训练记录

[返回首页](../README.md) · [结构神经 VAE](structure-vae.md)

本记录描述 2026-09-23 完成的数据源检索、样本筛选、双 VAE 训练、独立测试、重载审计和 AI-CAD 部署。公开仓库只保存处理代码、来源记录、汇总指标和哈希，不分发原始样本或模型权重。

## AnySearch 数据源选择

使用 AnySearch 3.1.1 执行通用搜索与 academic.dataset 垂直搜索，并提取可访问的官方页面。检索记录见 [deepcad-anysearch-source-selection.json](../benchmarks/deepcad-anysearch-source-selection.json)。

| 数据源 | 官方规模与表示 | 本次决定 |
| --- | --- | --- |
| [DeepCAD](https://www.cs.columbia.edu/cg/deepcad/) | 178,238 个 CAD 模型及构造序列，带官方划分 | 采用；与现有草图/拉伸重建、结构标签和 STEP/BREP 描述管线一致 |
| [Fusion 360 Gallery Reconstruction](https://github.com/AutodeskAILab/Fusion360GalleryDataset) | 8,625 条序列，下载约 2 GB | 暂缓；需要独立的 Fusion 360 转换链 |
| [SketchGraphs](https://github.com/PrincetonLIPS/SketchGraphs) | 1,500 万二维约束草图，序列文件约 15 GB | 暂缓；缺少当前几何 VAE 所需的已验证三维 STEP/BREP |
| [ABC Dataset](https://deep-geometry.github.io/abc-dataset/) | 100 万 B-Rep CAD 模型 | 暂缓；适合几何预训练，但没有当前结构 VAE 使用的构造序列标签 |

DeepCAD 官方下载入口与本机已有归档一致。本次复用该归档，记录归档 SHA-256、官方划分 SHA-256 和每个输入文档的 SHA-256。哈希用于复现实验输入，不能替代数据发布方签名或授权说明。DeepCAD 代码许可不能自动视为源 CAD 数据的再分发许可。

## 样本准备

选择种子为 20260923。从官方 train 随机选取 430 个候选，从官方 test 选取 100 个候选。每个样本必须满足：

1. 构造序列仅包含当前导入器支持的草图与拉伸操作；
2. 重建实体非空、体积为正且 BREP 有效；
3. 导出 STEP 后重新导入仍有效；
4. 44 维几何描述与已保留样本不重复。

| 阶段 | 数量 |
| --- | ---: |
| 候选样本 | 530 |
| 几何或 STEP 往返失败 | 6 |
| 重复描述向量剔除 | 19 |
| 最终训练域样本 | 415 |
| 最终独立测试样本 | 90 |
| 最终总数 | **505** |

去重使用保留八位小数的 44 维描述，优先保留训练记录；它不是完整拓扑同一性检测。结构标签来自操作、曲线类型及数量，不是自然语言标注。数据没有装配关节标注，不能据此评估装配语义或关节学习能力。

数据清单见 [deepcad-anysearch-505-manifest.json](../benchmarks/deepcad-anysearch-505-manifest.json)。

## 训练配置

官方 test 的 90 个样本不参与训练、归一化、词表构建、早停或权重选择。结构和几何模型均使用 415 个训练域样本，并在训练域内部保留约 20% 验证集。

| 参数 | 结构 VAE | 几何 VAE |
| --- | ---: | ---: |
| 输入 | 21 个结构标签 + 2 个数量特征 | 44 维 BREP 描述 |
| 隐藏层 | 128 | 64 |
| 潜空间 | 8 | 8 |
| 最大轮数 | 600 | 600 |
| 学习率 | 0.003 | 0.003 |
| KL 权重 | 0.01 | 0.01 |
| 预热轮数 | 20 | 20 |
| 批大小 | 32 | 64 |
| 固定种子 | 42 | 42 |

结构 VAE 以 50% 概率隐藏数量输入，模拟只有结构标签的查询；最佳权重位于第 87 轮，早停于第 147 轮。

## 独立测试结果

| 指标 | 神经 VAE | 对照 |
| --- | ---: | ---: |
| 几何标准化重建 MSE ↓ | 0.057888 | 训练均值预测：0.726326 |
| 结构标签 Brier 误差 ↓ | 0.004058 | 同输入 SVD：0.009994 |
| 数量标准化 log1p MSE ↓ | 0.020149 | 同输入 SVD：0.00003529 |
| 标签 Jaccard 最邻近一致率 ↑ | 92.22% | 同输入 SVD：91.11% |

与上一版 301 样本运行相比，标签 Brier 误差降低 39.35%，数量误差降低 81.94%，几何测试误差降低 59.24%；检索代理一致率下降 2.69 个百分点。神经模型在标签重建和检索代理上优于本次 SVD 对照，SVD 在数量重建上仍明显更好。

Jaccard 一致率使用结构标签最邻近样本作为透明代理，没有人工语义相关性标注，不能解释为自然语言检索准确率。完整结果见 [deepcad-anysearch-505-results.json](../benchmarks/deepcad-anysearch-505-results.json)。

## 审计与部署

505 个样本均完成结构与几何编码、解码。独立审计重新加载数据和模型，验证官方划分、来源文档哈希、训练索引、模型与数据哈希、有限输出和保存结果一致性，并复算三项独立测试指标。审计结果见 [deepcad-anysearch-505-audit.json](../benchmarks/deepcad-anysearch-505-audit.json)。

模型以 vae-deepcad-anysearch-505-20260923 部署到 AI-CAD。运行时只安装 415 个训练样本；90 个官方 test 样本保持隔离。真实 CadLatentTrainingManager.plannerModel() 加载与“拉伸 圆形草图 切除”检索验证通过。

通过审计表示数据、训练和推理链完整，不表示模型能直接生成 STEP，也不等于工程或制造认证。

## 复现命令

~~~bash
OPENBLAS_NUM_THREADS=1 python scripts/prepare_deepcad_training.py   --archive /path/to/cad_json.tar.gz   --split-file /path/to/train_val_test_split.json   --out-dir data/deepcad-anysearch-500-20260923   --train-count 430 --test-count 100 --seed 20260923

OPENBLAS_NUM_THREADS=1 python scripts/run_training_experiment.py   --data-dir data/deepcad-anysearch-500-20260923   --out-dir models/deepcad-anysearch-505-20260923   --structure-hidden-dim 128 --geometry-hidden-dim 64   --epochs 600 --patience 60 --seed 42

OPENBLAS_NUM_THREADS=1 python scripts/audit_training_run.py   --data-dir data/deepcad-anysearch-500-20260923   --model-dir models/deepcad-anysearch-505-20260923   --split-file /path/to/train_val_test_split.json
~~~

第三方库、平台和几何内核差异可能影响重建成功率及浮点结果。不要把不同归档或划分文件混入同一运行目录。
