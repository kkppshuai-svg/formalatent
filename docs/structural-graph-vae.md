# 结构引用图实验

这个实验研究草图轮廓与操作之间的引用。`deepcad_structure_graph.py` 从 DeepCAD 原始 JSON 提取草图、轮廓、拉伸操作、轮廓引用以及操作顺序。操作顺序不等同于布尔目标，因此导入器只记录数据明确提供的关系。原始尺寸和实体由原文档保留。

```powershell
python scripts/deepcad_structure_graph.py path/to/cad.json --out data/graph.json
```

`structural_graph_vae.py` 使用 PyTorch 多头注意力编码引用边，每个操作有独立的高斯潜表示。训练中将一条引用的潜表示从另一个样本交换进来，并用预期的新图进行监督。三种轮廓、两个操作的组合中，训练和测试组合严格分开。

```powershell
python -m pip install -r requirements-graph.txt
python scripts/structural_graph_vae.py --out models/graph-vae.pt
python scripts/structural_graph_vae.py --intervention-weight 0
```

安装 `requirements-cad.txt` 后增加 `--cad` 会执行底板切除并检查材料。可以提供 `{"base":[0,1],"source":[2,0],"slot":0}` 作为输入文件，运行以下命令进行干预。也可使用 `baseGraph` 和 `sourceGraph` 传入上一步导出的图。

```powershell
python scripts/structural_graph_cli.py --model models/graph-vae.pt --input examples/structural-graph-edit.json --cad
```

使用 `--profiles 4` 可以训练四轮廓模型，支持 3 至 16 个轮廓。模型要求输入的轮廓数量与训练时一致，每个操作引用一个轮廓。它只重建引用关系，输入中的节点和操作已知。`--cad` 始终检查预设底板的三个孔位。`--document 原文档.json --document-out 修改后.json --step-out 输出.step` 会把解码后的引用写回原文档并使用项目的 DeepCAD 导入器重放实体。重放支持草图和拉伸，每次 NewBody 建立独立实体。存在多个实体时，布尔操作的 targetBody 必须指向创建目标实体的拉伸操作 ID。只有一个实体时可以省略该字段。重放结果包含各实体的体积，每次操作必须产生一个有效且非空的实体。targetBody 由 CAD 文档提供，VAE 仍只修改轮廓引用。

模型没有学习通用的特征生成、几何参数、装配或任意 DeepCAD 模型。固定槽位与独立解码保证未替换槽位的输出不会受干预影响，对照实验应该据此解读。原始 DeepCAD 数据尚需另行提供，合成组合结果不能作为真实模型泛化指标。

本机另使用 [Autodesk Fusion 360 Gallery 的 Couch 示例](https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/tools/testdata/Couch.json) 验证文件解析和重放。提取得到 2 张草图、4 个轮廓和 2 次拉伸，四轮廓模型重建了两条引用。将第一条引用从索引 0 换到 2，第二条保持索引 1，两个实体均通过有效性检查。原实体体积约为 25.86，修改后约为 3.85。该例子只验证一次操作，不能说明修改符合设计意图。

`align_structure_latent.py` 可分析已有 `formalatent-structure-vae-v1` 模型。它训练一个方向以替换指定标签相关的潜表示，同时控制其他标签概率变化。训练对和测试对使用不同 JSONL 文件，每行包含 `base` 和 `source` 两个结构样本。

```powershell
python scripts/align_structure_latent.py --model models/structure-vae.json --train-pairs data/train-pairs.jsonl --test-pairs data/test-pairs.jsonl --token feature:hole --out models/hole-alignment.json
```

测试结果包含预测改变的样本数、目标标签正确率、其他标签概率变化和随机方向对照。该方法是单方向、基于交换的对齐实验，标签变化不自动证明真实 CAD 特征关系。使用独立零件族构造两份样本，并保证模型训练数据与最终测试数据隔离，才能评估泛化。

使用 `--document` 时，`structureExplanation` 列出改动前后的草图和轮廓 ID，并记录同一实体上随后重放的操作及其前后版本的体积。最终实体通过 CAD 布尔差集计算移除和新增的材料体积，即使净体积相同也能区分材料位置的变化。后续操作列表只表示这些操作作用于改过的实体，不能单凭这份列表断定每一步都产生了不同的几何。

## 本机实验记录

使用 AI-CAD 的 `cad-latent/training_samples.jsonl`，按样本分出 22 条训练数据和 6 条测试数据。用 22 条训练数据重新训练结构 VAE，再用其交叉组合训练方向。6 条测试数据形成 18 个有向配对，目标为 `feature:hole`。这 18 个配对的目标标签均正确，平均其他标签概率变化约 0.108，随机方向的目标标签正确率为 0。测试配对共享 6 个样本，因此这不是 18 个独立零件，样本也未按设计家族去重。

```powershell
python examples/prepare_alignment_pairs.py --dataset ../ai-cad/cad-latent/training_samples.jsonl --out-dir data/alignment
python scripts/train_cad_vae.py --dataset data/alignment/structure-train.jsonl --out models/alignment-structure.json
python scripts/align_structure_latent.py --model models/alignment-structure.json --train-pairs data/alignment/train-pairs.jsonl --test-pairs data/alignment/test-pairs.jsonl --token feature:hole
```
