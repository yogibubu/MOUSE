"""Read-only acoustic and rotational covariance checks of stored Hessians."""
from pathlib import Path
import os,socket,json,hashlib
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
W=Path('/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration')
rows=[]
manifest=W/'link_validation/accepted_run/multicsf_accepted/accepted.json'
record=manifest.parent/json.loads(manifest.read_text())['path']
for name,path in [('Furan',record.parent/'point.npz'),('Cyclopentadiene',W/'cpd_hessian/anchor/point.npz')]:
    with np.load(path) as a:x=a['coordinates_angstrom'].copy()*(1.0/0.529177210903);g=a['gradient'].reshape(-1,3).copy();h=a['hessian'].copy()
    x-=x.mean(axis=0);translations=[];rotations=[]
    for axis in np.eye(3):
        translations.append(float(np.max(abs(h@np.tile(axis,len(x))))))
        dx=np.cross(np.broadcast_to(axis,x.shape),x).ravel()
        expected=np.cross(np.broadcast_to(axis,g.shape),g).ravel()
        rotations.append(float(np.max(abs(h@dx-expected))))
    row=dict(molecule=name,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             hessian_symmetry_error=float(np.max(abs(h-h.T))),translation_residuals=translations,
             gradient_net_force=float(np.max(abs(g.sum(axis=0)))),rotation_covariance_residuals=rotations,
             gradient_net_torque=float(np.max(abs(np.cross(x,g).sum(axis=0)))))
    rows.append(row)
out=dict(host=socket.getfqdn(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,
         scope='Read-only Euclidean covariance identities on stored nonstationary-geometry analytic derivatives; H(rotation*R)=rotation*gradient, not zero rotational modes at a nonminimum')
folder=W/'rigid_body';folder.mkdir(exist_ok=True)
(folder/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
for r in rows:
    assert max(r['translation_residuals']+r['rotation_covariance_residuals'])<1e-7
    assert r['hessian_symmetry_error']<1e-10
