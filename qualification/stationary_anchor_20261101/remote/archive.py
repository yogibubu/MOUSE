"""Oracle inventory of preserved outputs, including untransferred grid caches."""
from anchor import *

if __name__=='__main__':
    files=[]
    for p in sorted(DEST.rglob('*')):
        if not p.is_file() or any(k in p.parts for k in ['__pycache__','.pytest_cache']):continue
        if p.name=='REMOTE_INVENTORY.json':continue
        files.append(dict(path=str(p.relative_to(DEST)),bytes=p.stat().st_size,sha256=sha(p)))
    inputs=[]
    for cohort,name in [('independent','Criegee_Intermediate'),('independent_challenge','Furofuran')]:
        base=OUT/cohort;pred=base/'predictions'/name;ref=base/'references'/name
        paths=[ref/'reference.npz',ref/'backend.chk',pred/'native_ground.npz',pred/'full_response/anchor/embedding.npz',pred/'full_response/anchor/projected_eri.npz']
        paths.extend(pred/profile/'anchor/model.npz' for profile in ['full_response','qualified_spectral_response'])
        rows=[r for r in json.loads((base/'manifest.json').read_text())['states'] if r['molecule']==name]
        paths.extend(ref/f'{r["id"]}.npz' for r in rows)
        inputs.extend(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths)
    write(DEST/'REMOTE_INVENTORY.json',dict(host=HOST,root=str(DEST),files=files,inputs=inputs,
        completed_checkpoints_preserved=True,large_quadrature_caches_retained_on_Oracle=True,source_sha256=sha(__file__)))
