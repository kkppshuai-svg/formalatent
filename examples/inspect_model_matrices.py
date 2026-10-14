"""Inspect actual trainable matrix shapes and small numeric excerpts from a saved VAE."""
import argparse
import json
from pathlib import Path
import numpy as np


def inspect(model):
    if model.get("format") not in {"ai-cad-brep-vae-v2", "formalatent-structure-vae-v1"}:
        raise ValueError("Expected a neural geometry or structure VAE, not an SVD baseline")
    d, h, latent = (model[key] for key in ("inputDim", "hiddenDim", "latentDim"))
    shapes = {"w1": (d, h), "b1": (h,), "w_mu": (h, latent), "b_mu": (latent,),
              "w_lv": (h, latent), "b_lv": (latent,), "w2": (latent, h), "b2": (h,),
              "w_out": (h, d), "b_out": (d,)}
    arrays, count = {}, 0
    for name, expected in shapes.items():
        value = np.asarray(model["weights"][name], dtype=np.float64)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"Invalid shape or nonfinite weight: {name}")
        preview = value[:2, :4] if value.ndim == 2 else value[:4]
        arrays[name] = {"shape": list(value.shape), "parameterCount": int(value.size),
                        "preview": preview.tolist()}
        count += value.size
    return {"kind": "saved-neural-model-matrices", "format": model["format"],
            "inputDim": d, "hiddenDim": h, "latentDim": latent,
            "trainableParameterCount": int(count), "arrays": arrays,
            "note": "Numeric previews are excerpts of the supplied model, not pretrained DeepCAD weights."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = inspect(json.loads(Path(args.model).read_text(encoding="utf-8")))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
