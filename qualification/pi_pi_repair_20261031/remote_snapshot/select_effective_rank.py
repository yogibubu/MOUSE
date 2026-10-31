"""Read fixed transition files only; do not read any excitation energies."""
from pathlib import Path
import socket,json,hashlib
import numpy as np
assert socket.getfqdn()=='oracle.sns.it'
V=Path('/home/enzo/MOUSE-orbital-performance-20261031');D=Path('/home/enzo/MOUSE-state-following-20261031/reduced_validation')
ids=['B001','B004','B005','B007','B011','B014','B015','B016','B022','B024','B025','B029','B030','B039'];rows=[]
for casefile in sorted((D/'prepared').glob('*/case.json')):
    case=json.loads(casefile.read_text())
    for target in case['targets']:
        if target['id'] not in ids:continue
        path=Path(target['transition_file'])
        with np.load(path) as a:sigma=np.linalg.svd(a['amplitude'],compute_uv=False)
        weights=sigma**2/np.sum(sigma**2);participation=float(1/np.sum(weights**2));rank=int(np.floor(participation+.5));assert 1<=rank<=4
        rows.append(dict(state=target['id'],molecule=casefile.parent.name,transition_file=str(path),transition_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),weights=weights.tolist(),participation=participation,rank=rank,distance_to_rank_boundary=float(.5-abs(participation-rank))))
assert {r['state'] for r in rows}==set(ids) and len(rows)==14
out=V/'pi_pi_repair/effective_rank_selection.json';out.write_text(json.dumps(dict(host=socket.getfqdn(),labels_read=False,rows=rows),indent=2)+'\n');print([(r['state'],round(r['participation'],4),r['rank']) for r in rows])
