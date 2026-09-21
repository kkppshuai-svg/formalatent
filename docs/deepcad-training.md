# DeepCAD 外部样本训练记录

[返回首页](../README.md) · [结构神经 VAE](structure-vae.md)

本次工作完成了来源查找、样本重建与筛选、两条 VAE 训练、独立测试及 AI-CAD 模型加载验证。审计完成于 2026-09-21，运行目录沿用启动时的标识 `deepcad-20260920`。

## 来源与数据取得

使用 AnySearch 查询 `DeepCAD CAD dataset official github cad_json license`，并提取[DeepCAD 官方仓库](https://github.com/rundiwu/DeepCAD)说明。官方提供原始 CAD 构建序列 JSON、向量数据和训练/验证/测试划分；论文为 [DeepCAD: A Deep Generative Network for Computer-Aided Design Models](https://arxiv.org/abs/2105.09492)。

官方说明中的[数据下载入口](http://www.cs.columbia.edu/cg/deepcad/data.tar)与本机已有归档的来源记录相符。本次复用已有归档，未重新下载；记录了归档、划分文件及每个导入 JSON 的 SHA-256。哈希用于追踪此次本地输入，不能替代数据发布方签名。AnySearch 的普通搜索与官方页面提取成功，学术数据垂直搜索出现 TLS 连接错误。

官方仓库代码提供 MIT 许可证；代码许可不能自动视为全部源 CAD 数据的再分发许可。本仓库只提交处理代码、汇总结果及哈希，不分发原始样本或训练权重。

来源摘要：[deepcad-source-manifest.json](../benchmarks/deepcad-source-manifest.json)。

## 样本准备

使用种子 `20260920`，从官方 train 中选取 256 个候选，从官方 test 中选取 64 个候选。逐个 JSON 重建 STEP；检查实体非空、体积为正、BREP 有效，且 STEP 导出再导入后仍有效。单个重建任务限制为 35 秒，最多同时处理两个样本。

| 阶段 | 数量 |
| --- | ---: |
| 候选样本 | 320 |
| 无效几何或 STEP 往返校验失败 | 3 |
| 重复描述向量剔除 | 16 |
| 最终训练样本 | 242 |
| 最终独立测试样本 | 59 |

去重使用保留八位小数的 44 维描述，优先保留训练记录；它不是完整拓扑同一性检测。当前导入器支持草图与拉伸构建序列，几何有效性检查不保证重建实体与原始 Onshape 模型完全一致。

结构标签来自序列中的操作、曲线类型与数量，不是人工撰写的自然语言说明。零件数取重建后的实体数；该数据没有装配关节标注，`jointCount=0` 只用于此单体 CAD 导入流程，不能据此评估关节学习能力。

## 训练与“通过 VAE”的定义

两条模型均使用 242 个训练域样本，内部保留约 20% 作为选取权重的验证集。官方 test 的 59 个样本不参与训练、归一化拟合或权重选择。固定训练种子为 `42`，最大训练轮数为 600。

- **几何 VAE**：44 维描述 → 8 维潜向量 → 44 维重建描述。
- **结构 VAE**：19 个结构标签及两个数量特征 → 8 维潜向量 → 标签概率及数量预测。

301 个样本均实际完成编码和解码，并保存为压缩 NumPy 文件。随后独立重新加载数据与模型，验证来源哈希、官方划分、索引只含训练域样本、输出有限且与保存结果一致，并重新计算几何测试误差。

通过的是**数据与训练推理流程的完整性验证**，不表示能够直接生成 STEP，也不等于获得工程或制造认证。审计结果：[structure-deepcad-audit.json](../benchmarks/structure-deepcad-audit.json)。

## 独立测试结果

| 指标 | 神经 VAE | 对照 |
| --- | ---: | ---: |
| 几何标准化重建 MSE（越低越好） | 0.142034 | 训练均值预测：0.881933 |
| 结构标签 Brier 误差（越低越好） | 0.062019 | 同特征 SVD：0.003205 |
| 数量标准化 log1p MSE（越低越好） | 0.165638 | 同特征 SVD：0.00000754 |
| 标签 Jaccard 最邻近一致率（越高越好） | 93.22% | 同特征 SVD：91.53% |

SVD 对照在与神经模型相同的无质量标签向量、训练子集、词表及数量归一化上拟合。检索一致率以标签 Jaccard 最大的训练样本作为代理参考，接受并列最大值；它没有人工语义相关性标注，不能代表自然语言查询准确率。

这次结构神经模型的重建明显弱于 SVD，检索代理指标略高。结果没有证明全面优于 SVD，因此保存对照、保留候选模型，不直接替换 AI-CAD 正在使用的权重。完整记录：[structure-deepcad-results.json](../benchmarks/structure-deepcad-results.json)。

## 复现命令

以下从已有官方归档开始，不包含下载步骤。需安装 `requirements-cad.txt`。

```bash
OPENBLAS_NUM_THREADS=1 python scripts/prepare_deepcad_training.py \
  --archive /path/to/cad_json.tar.gz \
  --split-file /path/to/train_val_test_split.json \
  --out-dir data/deepcad-20260920

OPENBLAS_NUM_THREADS=1 python scripts/run_training_experiment.py \
  --data-dir data/deepcad-20260920 --out-dir models/deepcad-20260920

OPENBLAS_NUM_THREADS=1 python scripts/audit_training_run.py \
  --data-dir data/deepcad-20260920 --model-dir models/deepcad-20260920 \
  --split-file /path/to/train_val_test_split.json
```

`prepare_deepcad_training.py` 默认候选数量与种子对应本报告。运行目录包含源 JSON、STEP、最终 JSONL、失败详情及来源清单；不要将不同输入归档混入同一运行目录。第三方库、平台差异可能影响几何重建和浮点结果。

训练产物包括 `structure-vae.json`、`geometry-vae.json`、`encoded-and-reconstructed.npz`、`report.json` 和 `audit.json`。旧格式 SVD 对照可单独生成：

```bash
python scripts/train_cad_vae.py --backend svd \
  --dataset data/deepcad-20260920/train.jsonl \
  --out models/deepcad-20260920/structure-svd-legacy.json
```

## 本机 AI-CAD 接入

本次产物另存于 AI-CAD 的 `pretraining-data/formalatent-neural-20260920/`，并部署了新格式的模型加载和检索代码。现有活动模型未被覆盖。

验证程序通过真实的 `CadLatentTrainingManager.plannerModel()` 加载新模型，并以“拉伸 圆形草图 切除”执行检索。验证使用临时 runtime 目录，避免影响现有会话：

```bash
node integrations/ai-cad/verify-model.mjs \
  /path/to/ai-cad /path/to/ai-cad/pretraining-data/formalatent-neural-20260920
```

源代码接入与运行中服务生效是不同步骤；长驻服务需重启后加载新代码。正式切换活动权重前，应使用目标领域的查询和几何样本评估候选模型。
