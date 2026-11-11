# 用 CAD 重放和运动结果训练连接排序

这项实验把 AI-CAD 装配方案转成结构图，遮掉一条驱动关节的子零件，让模型给其余候选端点排序。训练批次交错使用不同机构家族和已有 AI-CAD 案例。对照模型只学习正确端点，反馈模型还学习把正确端点排在可重放但运动错误的候选之前。

## 数据和验证

用 8 个机构模板各生成 3 个变体，包括铰链、直线滑轨、转台、嵌套滑台、摆臂载台、伸缩工具、支撑门和升降台。每个方案有一条被遮挡的驱动关节。变体共享母设计，留出评估时整个家族在同一折。模型只读零件角色、操作和连接图。目标运动的零件、轴、枢轴和行程只交给验证器。

零件使用 AI-CAD 的 CadQuery 构建器重建。候选先检查关节树和静止干涉，再采样 25%、50%、75% 和 100% 行程，检查指定零件的运动及实体干涉。本次 119 个候选中，24 个同时通过重放和运动，56 个通过重放但不满足运动目标，39 个关节树无效。这里的运动验证限于简化刚体和单个驱动关节，碰撞检查覆盖采样位置。

## 本机结果

每个种子用相同的 24 个生成方案做 8 折家族留出。下表是内核筛选前的正确端点数，分母均为 24。每折训练混入来自本机另外两个家族的 7 道原有结构题。

| 排序方法 | 首选正确，种子 42 / 7 / 123 | 前三命中，种子 42 / 7 / 123 |
| --- | --- | --- |
| 频次基线 | 9 / 9 / 9 | 23 / 23 / 23 |
| 结构图 VAE | 17 / 19 / 19 | 23 / 23 / 22 |
| 结构图 VAE 加运动反馈 | 19 / 17 / 19 | 24 / 23 / 22 |

从前三名依次选取第一个通过重放和运动检查的候选，三种排序方法的成功数分别与各自的前三命中数相同。每题只有一个候选通过验证。反馈损失在三个种子上对首选排序的影响不一致。候选数量为 3 至 8，前三命中也受候选集大小影响。

本机真实行星减速机家族全程留出。它有 3 个可测关节端点，频次基线首选命中 1 个，两种 VAE 在三个种子上均命中 3 个。这个结果只核对原方案中的端点，样本来自同一个机构，未提供本实验所需的独立运动目标。

## 复现

在同一工作区保留 `ai-cad` 和 `formalatent`，环境安装 PyTorch、CadQuery 和 Node.js。从工作区运行以下命令。

```powershell
node ai-cad/scripts/export-structure-graphs.mjs
Set-Location formalatent
$env:PYTHONPATH = "$PWD\data\torch-deps;$PWD\data\cad-deps"
python scripts/run_mechanism_feedback.py --out-dir data/mechanism-feedback --replay-dataset ../ai-cad/workspace/latent-learning/structure_graph_training.jsonl --real-test-family planetary-reducer
```

如果依赖已经装在当前 Python 环境，可以省略设置 `PYTHONPATH`。输出包含可重放的 `verified-plans.jsonl`、逐折候选排序与验证汇总 `report.json`，以及用所有通过验证的生成方案和旧回放数据训练的 `vae-with-feedback.pt`。权重用于离线实验。

新克隆的工作区也可以直接在 `formalatent` 目录运行 `python scripts/run_mechanism_feedback.py --out-dir data/mechanism-feedback`。这条命令用生成的机构方案训练权重并评估家族留出。Python 依赖见 [requirements-graph.txt](../requirements-graph.txt) 和 [requirements-cad.txt](../requirements-cad.txt)。

## 补全一个关节端点

输入 JSON 包含 `plan` 和 `probe`。在 `plan.joints` 中保留驱动关节的 `id`、`parent`、类型及其他参数，删除待补全的 `child`。在 `probe` 中用 `jointId` 指定该关节，用 `movingPart` 和 `stationaryParts` 指定应移动和应保持静止的零件。运动目标还需提供 `axis`、`pivot` 和非零 `travel`。零件的静止姿态与世界坐标轴对齐，轴和枢轴用世界坐标表示，旋转行程用弧度，直线行程用毫米。

在 `formalatent` 目录下，可以取训练产物中的第一条方案来试跑。

```powershell
@'
import json
from pathlib import Path

folder = Path("data/mechanism-feedback")
with (folder / "verified-plans.jsonl").open(encoding="utf-8") as rows:
    sample = json.loads(next(rows))
plan, probe = sample["plan"], sample["probe"]
joint = next(joint for joint in plan["joints"] if joint["id"] == probe["jointId"])
joint.pop("child", None)
draft = json.dumps({"plan": plan, "probe": probe}, ensure_ascii=False, indent=2)
(folder / "joint-draft.json").write_text(draft + "\n", encoding="utf-8")
'@ | python -X utf8 -

python scripts/complete_aicad_joint.py --model data/mechanism-feedback/vae-with-feedback.pt --input data/mechanism-feedback/joint-draft.json --plan-out data/mechanism-feedback/completed-plan.json
```

`--ai-cad` 可指定 AI-CAD 仓库路径，默认使用相邻的 `../ai-cad`。命令用模型对端点排序，以 CadQuery 重放排名最靠前的至多三个候选，检查关节树、静止干涉和四个行程采样位置的运动与干涉。它将第一个通过的方案写入 `--plan-out`。没有候选通过时，返回 `no_pass` 和退出码 2。此命令由用户离线调用，运动检查限于单个驱动关节、简化刚体和离散采样位置。
