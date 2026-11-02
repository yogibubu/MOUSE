"""Remote, independent nonstationary operator and trust-step controls."""
import socket,runpy,json
from pathlib import Path
import numpy as np
from pyscf import gto,dft
from threadpoolctl import threadpool_limits
assert socket.getfqdn().lower()=='mac-2.homenet.telecomitalia.it'
runpy.run_path('/Users/vincenzobarone/MOUSE-oo-response-20261101/source/qualification/remote_bootstrap.py')
from matrix_pyscf._oo_surface import OrbitalSurface
from matrix_pyscf._roks_optimizer import residual_trust_step
root=Path(__file__).resolve().parent
with threadpool_limits(2):
 mol=gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',basis='sto-3g',verbose=0)
 mf=dft.UKS(mol);mf.xc='PBE0';mf.grids.level=1
 with np.load('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-dft-roks-response-20261102/PBE0_checkpoint.npz') as z:c=z['coefficients'];occ=z['occupations']
 rng=np.random.default_rng(29);surface=OrbitalSurface(mf,c,occ,common=True)
 c=surface.rotate(c,.2*rng.normal(size=surface.dimension)/np.sqrt(surface.dimension))
 e,g,action,*unused=surface.evaluate(c)
 matrix=np.column_stack([action(v) for v in np.eye(len(g))]);v=rng.normal(size=len(g))
 error=float(np.max(abs(action.adjoint(v)-matrix.T@v)));assert error<1e-10,error
 assert np.max(abs(matrix-matrix.T))>1e-4
 step=1e-5;numeric=(surface.evaluate(surface.rotate(c,step*v),linearize=False)[1]-surface.evaluate(surface.rotate(c,-step*v),linearize=False)[1])/(2*step)
 de=float(np.max(abs(numeric-matrix@v)));assert de<1e-7,de
 newton=np.linalg.solve(matrix,-g);move,predicted,rank=residual_trust_step(g,action,newton,.03)
 assert np.linalg.norm(move)<=.03000000001 and predicted>0
 # Full-space least squares cannot be worse than a radial Newton clip in the
 # same trust ball; an explicit full solve is used only in this tiny control.
 clipped=newton*min(1.,.03/np.linalg.norm(newton))
 assert np.linalg.norm(g+matrix@move)<=np.linalg.norm(g+matrix@clipped)+1e-10
 count=surface.statistics['response_builds'];surface.evaluate(c,linearize=False)
 assert surface.statistics['response_builds']==count
 report=dict(host=socket.getfqdn(),exact_adjoint_error=error,moving_frame_FD_error=de,
             nonstationary_Jacobian_asymmetry=float(np.max(abs(matrix-matrix.T))),
             trust_radius=.03,step_norm=float(np.linalg.norm(move)),predicted_residual_reduction=predicted,
             trial_evaluation_response_builds=0)
 (root/'operators.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
