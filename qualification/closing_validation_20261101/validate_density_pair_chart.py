"""Full E/G/H equivalence under an invertible complete response-chart change."""
from pathlib import Path
import socket,runpy,json,shutil,hashlib
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
import numpy as np
from scipy.linalg import solve_sylvester
from dataclasses import replace
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.nosi_surface import PreparedNOSIModel,load_nosi_result,save_nosi_result
from matrix_gfn2.spin_reference import save_response_reference
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from threadpoolctl import threadpool_limits
with threadpool_limits(2):
 old=PreparedNOSIModel(HERE/'disentangled_bath_spectral_block_reference_eta_0.600000/closing_blind/Ethylene/model.json');n=len(old.anchor.overlap);nf=len(old.response.operators);rng=np.random.default_rng(39);R=np.eye(n)+.03*rng.normal(size=(n,n));ri=np.linalg.inv(R);i,j=np.tril_indices(n);w=np.where(i==j,1.,2**.5);E=np.zeros((nf,n,n));E[np.arange(nf),i,j]=1/w;E[np.arange(nf),j,i]=1/w;B=ri.T@E@ri;operators=np.array([2*solve_sylvester(old.anchor.overlap,old.anchor.overlap,b) for b in B]);T=np.linalg.lstsq(old.response.operators.reshape(nf,-1).T,operators.reshape(nf,-1).T,rcond=None)[0].T;inverse=np.linalg.inv(T);assert np.max(abs(T@old.response.operators.reshape(nf,-1)-operators.reshape(nf,-1)))<1e-11
 r=old.response;reference=replace(r,operators=operators,pair_charge_hartree=inverse.T@r.pair_charge_hartree@inverse,pair_spin_hartree=inverse.T@r.pair_spin_hartree@inverse,pair_reference_moments=T@r.pair_reference_moments,pair_linear_hartree=inverse.T@r.pair_linear_hartree,source_model=r.source_model+':equivalent_density_chart')
 out=HERE/'density_chart_validation';out.mkdir(exist_ok=True);save_response_reference(out/'response.npz',reference)
 for name in ['template.npz','arrays.npz']:shutil.copy2(old.root/name,out/name)
 m=dict(old.manifest);m['sha256']={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['response.npz','template.npz','arrays.npz']};(out/'model.json').write_text(json.dumps(m,indent=2)+'\n');new=PreparedNOSIModel(out/'model.json');calc=GFN2Calculator([{1:'H',6:'C'}[int(z)] for z in old.anchor.atomic_numbers],temperature_kelvin=0.);calc.evaluate=lambda *x,**k:(_ for _ in ()).throw(AssertionError('new SCC forbidden'));baseline=load_nosi_result(old.root/'analytic_audit/hessian.npz',old);result=new.evaluate(calc,old.anchor.coordinates_bohr/ANGSTROM_TO_BOHR,properties=('energy','gradient','hessian'));save_nosi_result(out/'hessian.npz',result)
 a,b=baseline.states[0],result.states[0];report=dict(host=socket.getfqdn(),energy_error_hartree=abs(a.energy_hartree-b.energy_hartree),gradient_max_error=float(np.max(abs(a.gradient_hartree_per_bohr-b.gradient_hartree_per_bohr))),Hessian_max_error=float(np.max(abs(a.hessian_hartree_per_bohr2-b.hessian_hartree_per_bohr2))),complete_pair_features=nf,orbitals_discarded=0,chart_condition=float(np.linalg.cond(T)));assert report['energy_error_hartree']<1e-10 and report['gradient_max_error']<1e-9 and report['Hessian_max_error']<1e-8;(out/'certificate.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
