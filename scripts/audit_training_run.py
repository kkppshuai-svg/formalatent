"""Independently reload datasets/models and verify the saved training artifacts."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
from structure_vae import StructureVAE, vectorize
from vae_runtime import VaeRuntime
from train_cad_vae import load_samples


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',required=True);ap.add_argument('--model-dir',required=True);ap.add_argument('--split-file',required=True)
    args=ap.parse_args();data=Path(args.data_dir);model_dir=Path(args.model_dir)
    manifest=json.loads((data/'manifest.json').read_text()); report=json.loads((model_dir/'report.json').read_text())
    official=json.loads(Path(args.split_file).read_text())
    check(hashlib.sha256(Path(args.split_file).read_bytes()).hexdigest()==manifest['splitSha256'],'split hash mismatch')
    records={k:load_samples(data/f'{k}.jsonl') for k in ('train','test')}
    seen=set()
    for split,rows in records.items():
        check(len(rows)==manifest['accepted'][split],'sample count mismatch')
        for r in rows:
            check(r['source']['split']==split and r['source']['deepcadId'] in official[split],'official split mismatch')
            check(r['id'] not in seen,'duplicate ID');seen.add(r['id'])
            doc=data/'documents'/(r['source']['deepcadId'].replace('/','_')+'.json')
            check(hashlib.sha256(doc.read_bytes()).hexdigest()==r['source']['documentSha256'],'source document modified')
            check(Path(r['geometry']['stepPath']).is_file(),'STEP file missing')
            check(r['geometry']['features']['topology.valid']==1,'invalid geometry')
    for name,expected in report['artifacts'].items():
        check(hashlib.sha256((model_dir/name).read_bytes()).hexdigest()==expected,f'artifact hash mismatch: {name}')
    for name,expected in report['datasetSha256'].items():
        check(hashlib.sha256((data/name).read_bytes()).hexdigest()==expected,'dataset hash mismatch')
    sr=StructureVAE(json.loads((model_dir/'structure-vae.json').read_text()));gr=VaeRuntime(json.loads((model_dir/'geometry-vae.json').read_text()))
    all_samples=records['train']+records['test']
    check([s['id'] for s in sr.model['samples']]==[s['id'] for s in records['train']],'structure index mismatch')
    check([s['id'] for s in gr.model['samples']]==[s['id'] for s in records['train']],'geometry index mismatch')
    sx=sr.encode(all_samples);sd=sr.decode(sx)
    gx=np.asarray([s['geometry']['vector'] for s in all_samples]);gz=gr.encode(gx);gd=gr.decode(gz)
    with np.load(model_dir/'encoded-and-reconstructed.npz',allow_pickle=False) as saved:
        for key,values in [('structureLatent',sx),('geometryLatent',gz),('geometryReconstruction',gd),('structureTokenProbabilities',sd['tokenProbabilities']),('structureCounts',sd['counts'])]:
            check(np.isfinite(values).all() and np.allclose(values,saved[key],atol=1e-10),f'reloaded inference mismatch: {key}')
    n=len(records['train']); test_mse=float(np.mean(((gd[n:]-gx[n:])/gr.std)**2))
    check(np.isclose(test_mse,report['geometry']['testNormalizedMse']),'reported geometry metric mismatch')
    structure_x=vectorize(records['test'],sr.vocabulary,sr.mean,sr.std); vocabulary_size=len(sr.vocabulary)
    token_brier=float(np.mean((sd['tokenProbabilities'][n:]-structure_x[:,:vocabulary_size])**2))
    normalized_counts=(np.log1p(sd['counts'][n:])-sr.mean)/sr.std
    count_mse=float(np.mean((normalized_counts-structure_x[:,vocabulary_size:])**2))
    check(np.isclose(token_brier,report['structure']['testTokenBrier']),'reported structure token metric mismatch')
    check(np.isclose(count_mse,report['structure']['testCountNormalizedLogMse']),'reported structure count metric mismatch')
    result={'passed':True,'verifiedSamples':len(all_samples),'trainSamples':n,'testSamples':len(all_samples)-n,'structureLatentShape':list(sx.shape),'geometryLatentShape':list(gz.shape),'testGeometryMseRecomputed':test_mse,'testStructureTokenBrierRecomputed':token_brier,'testStructureCountMseRecomputed':count_mse,'checks':['official split membership','source document checksums','dataset and model checksums','training-only retrieval indexes','reloaded encoding and reconstruction equality','finite outputs','recomputed held-out geometry MSE','recomputed held-out structure metrics'],'environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()}}
    (model_dir/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
