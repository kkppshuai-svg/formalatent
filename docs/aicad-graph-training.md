# AI-CAD 结构图训练

装配连接的候选排序与运动验证见[用 CAD 重放和运动结果训练连接排序](mechanism-feedback.md)。

数据来自 AI-CAD 已保存的原始建模方案和有来源的回流结构图。采集保留零件角色、启用的建模操作和顺序、显式特征依赖，以及装配关系和关节的有向零件端点。原生特征树和兼容字段同时存在时，优先采用原生树。训练输入不含 STEP、尺寸数值或来源路径。

训练时遮去一个操作类型、关系类型或连接目标，使用剩余的图预测它。目标预测给定源节点和连接类型，候选只包含可用节点，已知的同类连接从候选中移除。评估按设计家族留出，训练折内建立词表及频次基线。缺少该任务或正确类别的折记为未覆盖。当前模型用于离线验证。

## 本机结果

扫描到 53 份方案文件，按结构图精确去重后保留 5 张图。它们来自带孔安装板、手机支架和行星减速机 3 个家族。运行时训练集的另外 28 条记录只有摘要，无法恢复节点和边。导出清单保留每张图的源文件。

35 轮训练、3 折家族留出测试，种子 42 的结果如下。

| 遮挡内容 | 测试题数 | 训练折覆盖 | VAE 答对 | 频次基线答对 |
| --- | ---: | ---: | ---: | ---: |
| 基元类型 | 15 | 3 | 3 | 2 |
| 建模特征类型 | 13 | 13 | 8 | 7 |
| 特征依赖目标 | 3 | 0 | 0 | 0 |
| 装配关系类型与目标 | 12 | 0 | 0 | 0 |
| 关节类型与目标 | 6 | 0 | 0 | 0 |
| 合计 | 49 | 16 | 11 | 9 |

种子 7 和 123 分别答对 6 题和 4 题，频次基线仍为 9 题。49 道题来自同一批 5 张图。现有数据不足以确认稳定优势。关系、关节和可用于测试依赖目标的实例都来自行星减速机家族。留出该家族后，对应训练折缺少这些监督样本。逐题预测和覆盖标记保存在报告中。

## 复现

从包含 `ai-cad` 与 `formalatent` 的工作区运行。训练依赖见 `formalatent/requirements-graph.txt`。

```powershell
node ai-cad/scripts/export-structure-graphs.mjs
Set-Location formalatent
$env:PYTHONPATH = "$PWD\data\torch-deps"
python scripts/train_aicad_graph_vae.py --dataset ../ai-cad/workspace/latent-learning/structure_graph_training.jsonl --report data/aicad-graph/training-report.json --model models/aicad-graph-vae.pt
```

使用已安装 PyTorch 的环境时，可省略设置 `PYTHONPATH`。使用 `--seed 7` 或 `--seed 123` 可复查不同初始化的结果，输出路径可另行指定。权重使用全部 5 张图训练，报告使用各折独立训练的模型。两者均记录训练数据的 SHA-256。
