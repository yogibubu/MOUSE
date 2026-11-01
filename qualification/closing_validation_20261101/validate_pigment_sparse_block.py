"""Compare exact sparse spin recoupling with an archived large dense CI block."""
from pathlib import Path
import socket,runpy,json,time
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
import numpy as np
from scipy.linalg import eigh
from matrix_gfn2._nosi_canonical import block
from matrix_gfn2._nosi_configurations import basis,common_core_and_active
from threadpoolctl import threadpool_limits
with threadpool_limits(2):
 start=time.monotonic();folder=HERE/'pigment_cid_expansion/Beta_carotene/screening_8/frontier_8';report=json.loads((folder/'result.json').read_text());records=report['auxiliary_records'];rng=np.random.default_rng(912);ids=sorted(rng.choice(len(records),size=36,replace=False).tolist());selected=[records[i] for i in ids]
 with np.load(HERE/'pigment_models/Beta_carotene/state.npz') as z:s=z['overlap'].copy();c=z['coefficients'].copy()
 values,vectors=eigh(s);y=(vectors*values**.5)@vectors.T@c
 screen=HERE/'pigment_screening_expansion/Beta_carotene/frontier_8';h=np.load(screen/'h.npy');w=np.load(screen/'screened_pair.npy',mmap_mode='r').reshape((len(c),)*4);core,active=common_core_and_active(records)
 metric,matrix,details=block(basis(y,selected),y,h,w,core,active);indices=1+2*len(report['retained_ROKS_ordinals'])+np.array(ids)
 with np.load(folder/'matrices.npz') as z:oldh=z['h'][np.ix_(indices,indices)];olds=z['s'][np.ix_(indices,indices)]
 mask=~np.eye(len(ids),dtype=bool)
 for i,a in enumerate(selected):
  for j,b in enumerate(selected):
   if a['kind']==b['kind']=='four' and a['closed']==b['closed'] and a['open']==b['open']:mask[i,j]=False
 eh=float(np.max(abs(matrix-oldh)[mask]));es=float(np.max(abs(metric-olds)));assert eh<1e-8 and es<1e-10,(eh,es)
 result=dict(host=socket.getfqdn(),source_CSFs=report['basis_size'],sampled_CSFs=len(ids),sampled_offdiagonal_pairs=int(mask.sum()),Hamiltonian_max_error_hartree=eh,overlap_max_error=es,scope='Canonical off-diagonal block; native diagonal and four-spin functional overrides intentionally excluded',spin_transform='Exact sparse recoupling instead of dense multiplication; no configuration removed',wall_seconds=time.monotonic()-start,details=details)
 (HERE/'pigment_sparse_block_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
