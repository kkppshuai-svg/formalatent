"""Create a fixed-seed numerical dataset for the documented CLI walkthrough."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/demo")
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    vectors = np.random.default_rng(42).normal(size=(32, 3))
    records = [{"id": f"demo-{index}", "name": f"synthetic-{index}", "geometry": {
        "featureNames": ["demo.x", "demo.y", "demo.z"], "vector": vector.tolist()
    }} for index, vector in enumerate(vectors)]
    (output / "geometry.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    (output / "query.json").write_text(json.dumps(vectors[:2].tolist()) + "\n", encoding="utf-8")
    print(json.dumps({"samples": 32, "inputDimensions": 3, "seed": 42,
        "dataset": str(output / "geometry.jsonl"), "queries": str(output / "query.json"),
        "scope": "Synthetic numerical demonstration; not a CAD evaluation"}))


if __name__ == "__main__":
    main()
