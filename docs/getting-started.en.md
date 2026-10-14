# Getting started

[Overview](../README.md) · [中文指南](getting-started.md) · [Reference (中文)](reference.md)

This walkthrough trains and queries a small numerical model without STEP files. It verifies the CLI workflow, not CAD accuracy or unseen-part retrieval.

## Requirements and installation

Python 3.10+ and NumPy are sufficient for the core. CadQuery/OpenCascade is optional for STEP extraction. No GPU, AI credentials, or pretrained weights are required.

```bash
git clone https://github.com/kkppshuai-svg/formalatent.git
cd formalatent
```

Linux/Bash:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

The commands below assume `python` selects the configured environment. On PowerShell, you can replace it with `& ./.venv/Scripts/python.exe`.

## Prepare, train, and query

```bash
python examples/create_demo_dataset.py --out-dir data/demo
python scripts/train_brep_vae.py --dataset data/demo/geometry.jsonl --out models/demo.json --latent-dim 2 --hidden-dim 16 --epochs 100 --seed 42
python scripts/vae_cli.py encode --model models/demo.json --input data/demo/query.json --out data/demo/latents.json
python scripts/vae_cli.py decode --model models/demo.json --input data/demo/latents.json --out data/demo/reconstructed.json
python scripts/vae_cli.py search --model models/demo.json --input data/demo/query.json --limit 3
```

The generator creates 32 synthetic 3D records and two queries using seed 42. Encoding returns `(2, 2)` latent vectors; decoding returns `(2, 3)` descriptors. Search returns up to three matches per query. Lower mixed scores indicate nearer samples, not calibrated probabilities.

The queries come from the sample table. Identity matches do not measure generalization. See the [executed example summary](../examples/demo-result.json).

## STEP workflows

```bash
python -m pip install -r requirements-cad.txt
python scripts/extract_brep_geometry.py --step path/to/part.step
```

Prepare multiple samples, extract a geometry JSONL dataset, and train a model with the built-in 44-dimensional feature schema before running STEP queries. The synthetic 3D model is incompatible with STEP queries.

```bash
python scripts/query_brep_vae.py --model models/geometry-vae.json --step path/to/query.step --limit 5
```

Full dataset and fine-tuning instructions remain available in the [Chinese usage guide](usage.md). Feature names, order, and dimensions must agree between training and inference. Decode outputs are descriptors, not complete solids.

## Run checks

```bash
python -m pip install -r requirements-dev.txt
```

Bash:

```bash
PYTHONPATH=scripts python -m pytest scripts -q
```

PowerShell:

```powershell
$env:PYTHONPATH = "scripts"
python -m pytest scripts -q
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing NumPy or CadQuery | Install dependencies into the Python environment executing the command |
| Feature-schema mismatch | Match feature names, order, and dimensions; do not mix synthetic and STEP schemas |
| `vae_runtime` import fails | Set `PYTHONPATH=scripts` for Python integrations |
| Decode produces no STEP | Descriptor-to-solid generation is not implemented |
| Historical metrics cannot be reproduced | Raw data and weights are not distributed; prepare matching inputs using the [training record (中文)](deepcad-training.md) |

## Validation boundaries

Geometric and structural models are trained separately. Latent distances and reconstruction losses do not replace BRep, dimensional, assembly, or manufacturing checks. Recorded DeepCAD results use a fixed subset and tag-based retrieval proxies. See the [validation document (中文)](validation.md) for experiment boundaries and provenance.
