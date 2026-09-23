"""Train both neural VAEs on an official train split and evaluate untouched test IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from train_cad_vae import load_samples
from structure_vae import train, StructureVAE, vectorize
from train_brep_vae import train_brep_vae, write_model
from vae_runtime import VaeRuntime


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',required=True);ap.add_argument('--out-dir',required=True)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--epochs',type=int,default=600)
    ap.add_argument('--latent-dim',type=int,default=8)
    ap.add_argument('--structure-hidden-dim',type=int,default=128)
    ap.add_argument('--geometry-hidden-dim',type=int,default=64)
    ap.add_argument('--structure-learning-rate',type=float,default=0.003)
    ap.add_argument('--geometry-learning-rate',type=float,default=0.003)
    ap.add_argument('--beta',type=float,default=0.01)
    ap.add_argument('--structure-batch-size',type=int,default=32)
    ap.add_argument('--geometry-batch-size',type=int,default=64)
    ap.add_argument('--warmup-epochs',type=int,default=20)
    ap.add_argument('--patience',type=int,default=60)
    ap.add_argument('--count-mask-probability',type=float,default=0.5)
    args=ap.parse_args(); data=Path(args.data_dir); out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    training=load_samples(data/'train.jsonl'); test=load_samples(data/'test.jsonl')
    assert {s['id'] for s in training}.isdisjoint({s['id'] for s in test})
    structure=train(training,latent_dim=args.latent_dim,hidden_dim=args.structure_hidden_dim,
        epochs=args.epochs,learning_rate=args.structure_learning_rate,beta=args.beta,
        batch_size=args.structure_batch_size,seed=args.seed,warmup_epochs=args.warmup_epochs,
        patience=args.patience,count_mask_probability=args.count_mask_probability)
    write_model(structure,out/'structure-vae.json')
    names=training[0]['geometry']['featureNames']
    assert all(s['geometry']['featureNames']==names for s in training+test)
    matrix=np.array([s['geometry']['vector'] for s in training]); test_x=np.array([s['geometry']['vector'] for s in test])
    geometry=train_brep_vae(training,names,matrix,latent_dim=args.latent_dim,
        hidden_dim=args.geometry_hidden_dim,epochs=args.epochs,beta=args.beta,
        seed=args.seed,learning_rate=args.geometry_learning_rate,
        batch_size=args.geometry_batch_size,warmup_epochs=args.warmup_epochs)
    write_model(geometry,out/'geometry-vae.json')
    sr=StructureVAE(structure);gr=VaeRuntime(geometry)
    all_samples=training+test
    sz=sr.encode(all_samples); gz=gr.encode(np.vstack([matrix,test_x]))
    sd=sr.decode(sz); gd=gr.decode(gz)
    assert np.isfinite(sz).all() and np.isfinite(gz).all() and np.isfinite(gd).all() and np.isfinite(sd['tokenProbabilities']).all() and np.isfinite(sd['counts']).all()
    assert float(sz.std())>0 and float(gz.std())>0
    np.savez_compressed(out/'encoded-and-reconstructed.npz',structureLatent=sz,geometryLatent=gz,structureTokenProbabilities=sd['tokenProbabilities'],structureCounts=sd['counts'],geometryReconstruction=gd)
    x=vectorize(training,structure['vocabulary'],sr.mean,sr.std)
    tx=vectorize(test,structure['vocabulary'],sr.mean,sr.std)
    ti=structure['training']['trainIndices']; fit=x[ti]; mean=fit.mean(0)
    _,_,vh=np.linalg.svd(fit-mean,full_matrices=False); basis=vh[:structure['latentDim']]
    svd_rec=((tx-mean)@basis.T)@basis+mean
    n=len(training); vocab=len(sr.vocabulary)
    probs=sd['tokenProbabilities'][n:];counts=sd['counts'][n:]
    true_counts=np.array([[s['structure']['partCount'],s['structure']['jointCount']] for s in test])
    # Compare count reconstructions on log1p standardized scale, matching the structural objective.
    neural_counts=(np.log1p(counts)-sr.mean)/sr.std
    gv=(test_x-gr.mean)/gr.std; grec=(gd[n:]-gr.mean)/gr.std
    # Retrieval agreement uses structural token Jaccard as a transparent proxy, not human labels.
    def agreement(latents, train_latents):
        hits=0
        for i,s in enumerate(test):
            actual=int(np.argmin(np.linalg.norm(train_latents-latents[i],axis=1)))
            q=set(s['tokens']); overlaps=[len(q&set(t['tokens']))/max(1,len(q|set(t['tokens']))) for t in training]
            hits += overlaps[actual] >= max(overlaps)-1e-12
        return hits/len(test)
    report={'trainSamples':n,'testSamples':len(test),'testSplit':'official DeepCAD test; excluded from training and checkpoint selection','seed':args.seed,
       'runConfig':{'epochs':args.epochs,'latentDim':args.latent_dim,
         'structureHiddenDim':args.structure_hidden_dim,'geometryHiddenDim':args.geometry_hidden_dim,
         'structureLearningRate':args.structure_learning_rate,'geometryLearningRate':args.geometry_learning_rate,
         'beta':args.beta,'structureBatchSize':args.structure_batch_size,
         'geometryBatchSize':args.geometry_batch_size,'warmupEpochs':args.warmup_epochs,
         'earlyStoppingPatience':args.patience,'countMaskProbability':args.count_mask_probability},
       'structure':{'format':structure['format'],'vocabularySize':vocab,'training':structure['training'],'metrics':structure['metrics'],
         'testTokenBrier':float(np.mean((probs-tx[:,:vocab])**2)),'svdTestTokenBrier':float(np.mean((np.clip(svd_rec[:,:vocab],0,1)-tx[:,:vocab])**2)),
         'testCountNormalizedLogMse':float(np.mean((neural_counts-tx[:,vocab:])**2)),'svdTestCountNormalizedLogMse':float(np.mean((svd_rec[:,vocab:]-tx[:,vocab:])**2)),
         'testTokenJaccardNeighborAgreement':agreement(sz[n:],sz[:n]),'svdTestTokenJaccardNeighborAgreement':agreement((tx-mean)@basis.T,(x-mean)@basis.T),
         'testUnknownTokenFraction':sum(t not in sr.vocabulary for s in test for t in s['tokens'])/max(1,sum(len(s['tokens']) for s in test))},
       'geometry':{'format':geometry['format'],'metrics':geometry['metrics'],'testNormalizedMse':float(np.mean((grec-gv)**2)),'trainingMeanBaselineTestNormalizedMse':float(np.mean(gv**2))},
       'gates':{'disjointOfficialIds':True,'allInputsEncodedAndDecoded':True,'finiteOutputs':True,'nonconstantLatents':True},
       'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'structure-vae.json',out/'geometry-vae.json',out/'encoded-and-reconstructed.npz')},
       'datasetSha256':{name:hashlib.sha256((data/name).read_bytes()).hexdigest() for name in ('train.jsonl','test.jsonl')},
       'limitations':['Small deterministic subset, no human semantic relevance labels','Structure input is unordered CAD tags/counts, not natural-language understanding or sequence generation','SVD comparison uses identical quality-free vectors and train-only transforms; legacy SVD is still available','No guarantee of geometric fidelity to original Onshape source; supported-sequence rebuild only','Passing gates establishes ingestion/inference integrity, not CAD generation or manufacturing readiness']}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({k:report[k] for k in ('trainSamples','testSamples','gates','geometry')},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
