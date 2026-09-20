# aicad-vae — AI-CAD 的 VAE

AI-CAD 几何描述向量的 VAE 编码、解码、训练与检索工具，独立于网页服务运行。
来源：`kkppshuai-svg/ai-cad`，基线提交 `52f44be`。本次实现版本 3.0.0，保留 `ai-cad-brep-vae-v2` 权重格式兼容性。

## 升级内容

- 编码/解码：可复用 NumPy 运行时缓存权重，批量编码、后验均值/方差、解码、重建，输入和权重形状/有限值校验。
- 训练：归一化只使用训练集；微调保留完整权重及原归一化坐标系；小批量 Adam、KL warmup、验证集早停；保存训练/验证索引，支持复现。
- 检索：批量查询及向量化距离计算；新模型结合几何描述距离与 latent 距离，减少压缩后相近 latent 的歧义；旧模型自动回退 latent 检索。
- 性能：同一进程重复查询时只加载一次模型，批量矩阵运算；逐查询距离计算避免创建查询数 × 样本数 × 特征数的大型张量。

这里的解码输出是 **44 维几何统计描述**，不是可直接制造的 STEP/BREP 实体。文本结构管线仍是明确标注的线性基线，并非神经 VAE。本仓库不包含私人建模记录、API 密钥、原始训练集或预训练权重。

## 安装与使用

Python 3.10+：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# 需要从 STEP 提取特征时：
pip install -r requirements-cad.txt
```

几何数据为 JSONL，每行包含 `id`、可选 `name`、`geometry.featureNames` 与 `geometry.vector`。所有记录必须使用相同的特征顺序，至少两条；建议准备足够的独立训练与验证样本。

```bash
python scripts/train_brep_vae.py --dataset data/geometry.jsonl --out models/vae.json --batch-size 64 --warmup-epochs 20
python scripts/train_brep_vae.py --dataset data/finetune.jsonl --init-model models/vae.json --out models/tuned.json --learning-rate 0.003
python scripts/query_brep_vae.py --model models/vae.json --step part.step
```

批量 JSON 输入为二维数组：

```bash
python scripts/vae_cli.py encode --model models/vae.json --input vectors.json --out latents.json
python scripts/vae_cli.py decode --model models/vae.json --input latents.json --out reconstructed.json
python scripts/vae_cli.py search --model models/vae.json --input vectors.json --limit 5
```

进程内高频调用：

```python
# 将 scripts 加入 PYTHONPATH
from train_brep_vae import load_model
from vae_runtime import VaeRuntime
runtime = VaeRuntime(load_model('models/vae.json'))
z = runtime.encode(vectors)
mu, log_variance = runtime.encode(vectors, posterior=True)
reconstructed = runtime.decode(z)
matches = runtime.search(vectors, limit=5, geometry_weight=0.5)
```

## 验证与基准

```bash
pip install -r requirements-dev.txt
PYTHONPATH=scripts python -m pytest scripts -q
OPENBLAS_NUM_THREADS=1 python benchmarks/evaluate.py --baseline /path/to/original/train_brep_vae.py --dataset data/geometry.jsonl --out benchmarks/results.json
```

`benchmarks/*results.json` 记录本机实验。重建比较使用相同验证记录的原始描述 MSE；旧版归一化存在验证信息泄漏，不能把旧版 loss 与新版 loss 直接比较。检索指标仅为 3% 描述噪声下的样本身份命中率，不代表真实用户查询的语义准确率。编码加速比较“重复旧版单条调用”与“复用运行时批量调用”，不代表模型训练或端到端 STEP 处理同等倍数提速。

旧版 JSON 权重可直接加载。新模型仍写入兼容 v2 格式，新增字段为可选元数据；混合检索需要新模型的 `geometryVector`，旧模型自动使用 latent。升级微调不会重新计算归一化，如目标域差异很大应另训模型并独立评估。
