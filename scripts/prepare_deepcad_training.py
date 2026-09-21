"""Import a bounded official DeepCAD split with isolated, time-limited CAD builds."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path


def worker(document_path, step_path):
    from import_deepcad_subset import build_shape_from_document
    from extract_brep_geometry import extract_step_features
    import cadquery as cq
    doc = json.loads(Path(document_path).read_text())
    supported = {'Sketch','ExtrudeFeature'}
    if any(item.get('type') not in supported for item in doc['sequence']):
        raise ValueError('unsupported operation in sequence')
    shape = build_shape_from_document(doc)
    if not shape.isValid() or not shape.Solids() or shape.Volume() <= 0:
        raise ValueError('invalid or empty BREP')
    cq.exporters.export(shape,str(step_path))
    geo = extract_step_features(step_path)
    if geo['features']['topology.valid'] != 1:
        raise ValueError('STEP roundtrip produced invalid geometry')
    types = Counter(item['type'] for item in doc['sequence'])
    curves = Counter()
    operations = Counter()
    for entity in doc['entities'].values():
        if entity.get('type') == 'Sketch':
            for profile in entity.get('profiles',{}).values():
                for loop in profile.get('loops',[]):
                    curves.update(c['type'] for c in loop.get('profile_curves',[]))
        if entity.get('type') == 'ExtrudeFeature':
            operations[entity.get('operation','unknown')] += 1
    tok = ['feature:extrude']
    tok += [f'operation:{k}' for k in operations]
    tok += [f'curve:{k}' for k in curves]
    tok += [f'sequence:{k}:{v}' for k,v in types.items()]
    return {'tokens':sorted(tok),'structure':{'partCount':geo['facts']['solidCount'],'jointCount':0,'featureTypes':dict(types)},'geometry':geo}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive'); ap.add_argument('--split-file'); ap.add_argument('--out-dir')
    ap.add_argument('--train-count',type=int,default=256); ap.add_argument('--test-count',type=int,default=64)
    ap.add_argument('--seed',type=int,default=20260920)
    ap.add_argument('--worker',nargs=2)
    args=ap.parse_args()
    if args.worker:
        print(json.dumps(worker(*args.worker))); return
    if not all((args.archive,args.split_file,args.out_dir)):
        ap.error('archive, split-file and out-dir are required')
    from random import Random
    out=Path(args.out_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    (out/'documents').mkdir(exist_ok=True); (out/'step').mkdir(exist_ok=True)
    splits=json.loads(Path(args.split_file).read_text()); jobs=[]
    # Stable and disjoint official IDs; no selection based on test outcomes.
    for split,count in [('train',args.train_count),('test',args.test_count)]:
        ids=list(splits[split]); Random(args.seed).shuffle(ids)
        jobs.extend((split,i) for i in ids[:count])
    assert len({i for _,i in jobs})==len(jobs)
    selected = {f'cad_json/{i}.json': i for _,i in jobs}
    found=set()
    with tarfile.open(args.archive,'r|gz') as archive:
        for member in archive:
            if member.name in selected:
                i=selected[member.name]
                raw=archive.extractfile(member).read()
                (out/'documents'/f'{i.replace("/","_")}.json').write_bytes(raw)
                found.add(member.name)
    if found != set(selected):
        raise ValueError('archive is missing selected official IDs')
    print(json.dumps({'archiveScanComplete':len(found)}),flush=True)
    def run(job):
        split,i=job; name=i.replace('/','_'); doc=out/'documents'/f'{name}.json'
        try:
            result=subprocess.run([sys.executable,__file__,'--worker',str(doc),str(out/'step'/f'{name}.step')],capture_output=True,text=True,timeout=35)
            if result.returncode: raise ValueError(result.stderr[-1200:])
            record=json.loads(result.stdout)
            record.update(id=f'deepcad-{name}',name=f'DeepCAD {i}',source={'dataset':'DeepCAD','split':split,'deepcadId':i,'documentSha256':hashlib.sha256(doc.read_bytes()).hexdigest()})
            return split,record,None
        except Exception as exc:
            return split,None,{'id':i,'error':str(exc)}
    records={'train':[],'test':[]}; failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for n,(split,record,error) in enumerate(pool.map(run,jobs),1):
            if record: records[split].append(record)
            else: failures.append({'split':split,**error})
            if n%16==0: print(json.dumps({'processed':n,'total':len(jobs),'accepted':sum(map(len,records.values()))}),flush=True)
    # Remove exact descriptor duplicates across train/test, preserving train.
    seen=set(); duplicates=[]
    for split in ('train','test'):
        clean=[]
        for r in records[split]:
            fingerprint=tuple(round(v,8) for v in r['geometry']['vector'])
            if fingerprint in seen:
                duplicates.append({'split':split,'id':r['id']}); continue
            seen.add(fingerprint); clean.append(r)
        records[split]=clean
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in clean))
    manifest={'source':'https://github.com/rundiwu/DeepCAD','downloadUrl':'http://www.cs.columbia.edu/cg/deepcad/data.tar','acquisition':'existing local archive; official source rediscovered via AnySearch','archiveSha256':hashlib.sha256(Path(args.archive).read_bytes()).hexdigest(),'splitSha256':hashlib.sha256(Path(args.split_file).read_bytes()).hexdigest(),'seed':args.seed,'selected':len(jobs),'accepted':{k:len(v) for k,v in records.items()},'failures':failures,'descriptorDuplicatesRemoved':duplicates}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'completed':True,'accepted':manifest['accepted'],'failures':len(failures),'duplicates':len(duplicates)}))

if __name__=='__main__': main()
