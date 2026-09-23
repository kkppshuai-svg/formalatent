# 文本结构神经 VAE

[返回首页](../README.md)

## 模型与边界

`formalatent-structure-vae-v1` 是基于全连接网络的 VAE。编码器输出高斯后验的均值及对数方差，训练时使用重参数采样，解码器同时预测结构标签与零件/关节数量。实现采用 NumPy，梯度通过有限差分测试核验。

输入是无序的 CAD 标签和数量特征，不是完整的自然语言或建模操作序列。模型不能直接生成参数化 CAD 程序、装配图或 STEP 实体。

训练目标为标签的平均二元交叉熵、两个标准化数量特征的平均 MSE 与加权 KL 项之和。数量先经 `log1p` 变换，再使用训练子集的均值与标准差归一化。词表也仅从训练子集构建，未知标签被忽略。

质量评分不进入编码向量；`quality:`、`freecad:`、`cadquery:` 标签被排除。评分保留为样本元数据，在检索排序阶段使用。训练时对一半样本隐藏数量输入，用于模拟只提供文字标签的查询；缺失数量输入采用归一化后的零，即训练均值。

## 训练与 SVD 对照

```bash
python scripts/train_cad_vae.py --dataset data/structure.jsonl --out models/structure-vae.json --backend neural --epochs 400
python scripts/train_cad_vae.py --dataset data/structure.jsonl --out models/structure-svd.json --backend svd
```

默认 backend 为 `neural`，至少需要三个样本。SVD 保留旧格式和原有质量特征，用于兼容；实验脚本另提供使用相同无质量特征的公平 SVD 对照。

输入 JSONL 示例：

```json
{"id":"part-1","tokens":["feature:hole","zh:支架"],"structure":{"partCount":1,"jointCount":0},"quality":{"score":100}}
```

神经训练参数包括 `--latent-dim 8`、`--hidden-dim 128`、`--epochs 400`、`--learning-rate 0.003`、`--beta 0.01`、`--batch-size 32`、`--seed 42`、`--warmup-epochs 20`、`--patience 60` 和 `--count-mask-probability 0.5`。约 20% 的记录用于验证及选取权重，不参与词表与数量统计拟合；模型记录最优轮数、实际轮数、早停耐心值、数量遮蔽概率和划分索引。

数量遮蔽用于模拟只有文本标签、没有零件或关节数量的查询。取值必须在 `[0,1]`；`0` 表示训练时始终提供数量，`1` 表示始终隐藏数量。2026-09-23 的外部样本训练保持 `0.5`，只将隐藏层从 64 扩展到 128，并依据训练域内部验证集选择权重。

## 编码、解码与检索

```bash
python scripts/query_cad_vae.py --model models/structure-vae.json --query '支架 安装孔' --limit 5
python scripts/structure_vae_cli.py encode --model models/structure-vae.json --input data/queries.json --out data/structure-latents.json
python scripts/structure_vae_cli.py decode --model models/structure-vae.json --input data/structure-latents.json
```

`queries.json` 是样本对象数组，例如 `[{"tokens":["feature:hole"]}]`。编码输出二维潜向量；解码返回有序词表、逐标签概率和两个数量预测。数量预测非负但不取整；对数值裁剪到 `[0,20]`，不保证组合符合真实装配约束。`reconstruct` 可直接重建样本数组。

Python 接口（`PYTHONPATH=scripts`）：

```python
from structure_vae import StructureVAE
model = StructureVAE(payload)  # payload 为读取后的模型 JSON
mu, logvar = model.encode([{"tokens": ["feature:hole"]}], posterior=True)
result = model.decode(mu)
```

文字查询仍采用规则分词与标签匹配，排序结合标签分数、质量分数和潜空间距离。旧 SVD 模型继续由同一查询脚本支持；神经模型不能传给几何模型的 `VaeRuntime`。

## AI-CAD 接入

`integrations/ai-cad/structure-vae-runtime.mjs` 提供与 Python 编码器数值一致的 JS 后验均值推理；`neural-structure.patch` 更新 AI-CAD 的模型加载与检索分支。接入时需同时部署 Python 脚本、JS 模块和补丁，并在服务重启后使用新格式模型。

新模型的最低训练样本数为三；训练管理器使用该下限，并将配置的训练轮数同时传给两条神经 VAE 管线。旧模型与模型文件不应被静默迁移；应保留备份并验证当前部署版本能够读取新格式。
