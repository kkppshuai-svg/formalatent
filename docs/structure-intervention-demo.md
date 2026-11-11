# 双轮廓结构干预实验

这个实验训练项目现有的 NumPy 结构 VAE，输入包含底板、两个圆形轮廓和一个切除操作。两个训练样例的唯一区别是切除引用左轮廓还是右轮廓。模型解码两种引用概率，并输出对应的结构图。

```bash
python scripts/structure_intervention_demo.py
```

有 CadQuery 环境时可以运行内核验证：

```bash
python -m pip install -r requirements-cad.txt
python scripts/structure_intervention_demo.py --cad --model-out models/two-profile-demo.json
```

`--cad` 分别创建底板和两个切削圆柱，检查切除前后的实体有效性、被切掉的体积，以及两个位置是否各自保留预期的材料。模型和测试结果可以通过命令行保存或重定向。

实验使用合成样本和一个潜变量。交换该潜变量会得到来源样本的完整潜表示，实验适合验证训练、解码、干预与内核验证的接口，不能证明模型能在复杂零件中独立修改一条依赖。结构图的固定节点由实验代码提供，模型预测的是切除引用。实验没有实现 DAG Transformer、DAS 或交换干预训练，也没有使用独立零件族进行泛化评估。
