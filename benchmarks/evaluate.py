"""Compare the upstream trainer and loop search against this release on local data."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from train_brep_vae import train_brep_vae, load_geometry_dataset
from vae_runtime import VaeRuntime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location('baseline', args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    records, names, matrix = load_geometry_dataset(args.dataset)
    reports = []
    for seed in (11, 42, 73):
        start = time.perf_counter()
        old = baseline.train_brep_vae(records, names, matrix, epochs=400, seed=seed)
        old_time = time.perf_counter()-start
        start = time.perf_counter()
        new = train_brep_vae(records, names, matrix, epochs=400, seed=seed)
        new_time = time.perf_counter()-start
        # Same split; compare raw descriptor reconstruction rather than differently normalized losses.
        validation = matrix[new['training']['validationIndices']]
        old_runtime, runtime = VaeRuntime(old), VaeRuntime(new)
        old_mse = float(np.mean((old_runtime.reconstruct(validation)-validation)**2))
        new_mse = float(np.mean((runtime.reconstruct(validation)-validation)**2))
        # Noisy held-out query observations, identity retrieval; not a semantic relevance claim.
        query = matrix + np.random.default_rng(seed).normal(size=matrix.shape)*np.maximum(matrix.std(axis=0), 1e-6)*0.03
        old_hits = old_runtime.search(query, limit=1, geometry_weight=0)
        new_hits = runtime.search(query, limit=1)
        recall = lambda hits: sum(str(hit[0]['id']) == str(r['id']) for hit, r in zip(hits, records))/len(records)
        repeats = 100
        start = time.perf_counter()
        for _ in range(repeats):
            for row in query:
                baseline.encode_vectors(new, [row])
        old_encode = time.perf_counter()-start
        start = time.perf_counter()
        for _ in range(repeats):
            runtime.encode(query)
        new_encode = time.perf_counter()-start
        reports.append(dict(seed=seed, baselineValidationRawMse=old_mse, upgradedValidationRawMse=new_mse, baselineNoisyIdentityRecallAt1=recall(old_hits), upgradedNoisyIdentityRecallAt1=recall(new_hits), baselineTrainSeconds=old_time, upgradedTrainSeconds=new_time, batchEncodeSpeedup=old_encode/new_encode))
    result = {'sampleCount': len(records), 'features': len(names), 'method': 'Three fixed seeds; identical validation IDs; raw descriptor MSE. Baseline normalization includes validation data. Retrieval queries are 3% descriptor noise, not independent semantic labels. Timing is local, warmed NumPy; batching vs repeated single-row legacy calls.', 'results': reports}
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
