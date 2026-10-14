# Technology stack and module map

[Overview](../README.md) · [中文](tech-stack.md) · [Neural matrices](neural-network.en.md)

## Dependency layers

| Layer | Technology and constraint | Source | Responsibility |
| --- | --- | --- | --- |
| Standalone core | Python 3.10+, NumPy ≥1.24,<3 | [requirements.txt](../requirements.txt) | Numerical training, encoding, decoding, and retrieval |
| Geometry extraction | CadQuery ≥2.4,<3 / OpenCascade | [requirements-cad.txt](../requirements-cad.txt) | STEP/BRep ingestion and geometric statistics |
| Development checks | pytest ≥7,<10 plus core/CAD dependencies | [requirements-dev.txt](../requirements-dev.txt) | Training, data, geometry, and runtime regression checks |
| AI-CAD integration | JavaScript ES modules / Node.js | [Integration directory](../integrations/ai-cad) | Structural posterior-mean inference, verification, and integration patch |

Numerical training and queries use only Python and NumPy. Install CAD dependencies for STEP inputs. OpenCascade is supplied through CadQuery's geometry bindings; this repository does not separately pin OCCT. The development requirements include the CAD dependencies.

JavaScript is an optional workbench integration layer. Its runtime computes deterministic structural posterior means from tags; it is not the Python trainer or the geometry batch runtime.

## Neural implementation

The VAEs use NumPy arrays and matrix multiplication, five dense weight matrices, five bias vectors, tanh activations, and Gaussian posterior branches. Training uses reparameterized sampling, explicit chain-rule gradients, and Adam updates.

These are trainable neural networks implemented in NumPy. The core requirements declare no additional deep-learning framework. SVD is a separate non-neural baseline. See [matrix operations](neural-network.en.md) for actual shapes, parameter counts, equations, and saved-model inspection.

## Source map

| Module | Responsibility |
| --- | --- |
| [train_brep_vae.py](../scripts/train_brep_vae.py) | Geometry VAE initialization, forward pass, gradients, Adam, training, and persistence |
| [structure_vae.py](../scripts/structure_vae.py) | Tag/count inputs, structure objectives, training, and inference |
| [vae_runtime.py](../scripts/vae_runtime.py) | Validated cached geometry arrays and batched inference/retrieval |
| [vae_cli.py](../scripts/vae_cli.py) | argparse CLI and JSON I/O |
| [extract_brep_geometry.py](../scripts/extract_brep_geometry.py) | STEP descriptors and JSONL datasets |
| [prepare_deepcad_training.py](../scripts/prepare_deepcad_training.py) | Supported DeepCAD sequence rebuild, filtering, and provenance |
| [run_training_experiment.py](../scripts/run_training_experiment.py) | Dual-model experiments, NPZ outputs, and held-out metrics |
| [structure-vae-runtime.mjs](../integrations/ai-cad/structure-vae-runtime.mjs) | JavaScript structural posterior-mean inference |
| [verify-model.mjs](../integrations/ai-cad/verify-model.mjs) | Workbench integration verification |

## Formats and project boundaries

Samples use UTF-8 JSONL. Model weights and metadata use JSON; experiment arrays can use compressed NumPy NPZ. Trainable matrices, normalization statistics, vocabularies, sample indexes, and evaluation metrics serve separate roles.

DeepCAD is an experiment data source, not a required runtime component. Python, model formats, and experiment inputs have independent versions.

FormaLatent runs independently and includes AI-CAD integration modules. STEP extraction uses CadQuery/OpenCascade. There is currently no native NVIDIA USD BRep feature adapter. [USD BRep Review](https://github.com/kkppshuai-svg/usd-brep-review) is a separate review layer, not a core dependency of this repository.
