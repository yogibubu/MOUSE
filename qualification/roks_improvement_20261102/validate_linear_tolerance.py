"""Controlled same-root robustness/cost comparison against committed baseline."""
import socket,runpy,json,types,sys,time
from pathlib import Path
import numpy as np
from pyscf import gto,dft
from threadpoolctl import threadpool_limits
assert socket.getfqdn().lower()=='mac-2.homenet.telecomitalia.it'
base=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');root=Path(__file__).resolve().parent
archive=root;root=root/'final_solver_linear_tolerance';root.mkdir(exist_ok=True)
runpy.run_path(str(base/'source/qualification/remote_bootstrap.py'))
from matrix_pyscf import optimize_roks,PySCFROKSConvergenceError
import matrix_pyscf.deltascf
from matrix_pyscf._oo_surface import OrbitalSurface
from matrix_pyscf.roks import _singlet_overlap
package=types.ModuleType('baseline_roks');package.__path__=[str(archive/'baseline_sources')];sys.modules[package.__name__]=package
sys.modules['baseline_roks.deltascf']=matrix_pyscf.deltascf
import baseline_roks.roks as baseline
import baseline_roks._oo_surface as baseline_surface
original=baseline_surface.OrbitalSurface
class CountingSurface(original):
 counts=None
 def evaluate(self,c):
  self.counts['Fock_builds']+=2;self.counts['response_builds']+=2
  e,g,action,*tail=super().evaluate(c)
  def counted(v):self.counts['Jacobian_actions']+=1;return action(v)
  return e,g,counted,*tail
baseline_surface.OrbitalSurface=CountingSurface
cases=[('water','sto-3g','PBE0_checkpoint.npz','O 0 0 0; H 0 -.757 .587; H 0 .757 .587',1),
       ('formaldehyde','6-31g','formaldehyde_PBE0_roks_checkpoint.npz','C 0 0 0; O 0 0 1.21; H 0 .94 -.59; H 0 -.94 -.59',2)]
rows=[]
with threadpool_limits(2):
 for name,basis,filename,atoms,grid in cases:
  mol=gto.M(atom=atoms,basis=basis,verbose=0);mf=dft.UKS(mol);mf.xc='PBE0';mf.grids.level=grid;mf.small_rho_cutoff=0
  with np.load(base/'oracle/MOUSE-dft-roks-response-20261102'/filename) as z:reference=z['coefficients'];occ=z['occupations']
  surface=OrbitalSurface(mf,reference,occ,common=True)
  for amplitude in [.15,.45,.8]:
   caseid=name+'_'+str(amplitude);path=root/(caseid+'_retained_target_solver.json')
   if path.exists():rows.append(json.loads(path.read_text()));continue
   rng=np.random.default_rng(73);direction=rng.normal(size=surface.dimension);direction*=amplitude/np.linalg.norm(direction)
   initial=surface.rotate(reference,direction);result_rows=[]
   for label,solver in [('trust',optimize_roks)]:
    counts=dict(Fock_builds=0,response_builds=0,Jacobian_actions=0);CountingSurface.counts=counts
    start=time.monotonic()
    # Supply the same retained physical target to both algorithms. The baseline
    # has no target argument, so its existing overlap callback is rebound only.
    original_overlap=baseline._singlet_overlap
    if label=='baseline':baseline._singlet_overlap=lambda ignored,c,o,s:_singlet_overlap(reference,c,o,s)
    options=dict(reference_coefficients=reference) if label=='trust' else {}
    try:r=solver(mf,initial,occ,orbital_tolerance=1e-9,maximum_iterations=100,strict=False,**options)
    finally:baseline._singlet_overlap=original_overlap
    score=_singlet_overlap(reference,r.common_coefficients,occ,mf.get_ovlp())
    if label=='trust':counts={k:r.solver_certificate[k] for k in counts}
    result_rows.append(dict(solver=label,converged=r.converged,correct_root=score>.9,CSF_overlap=score,
       energy_hartree=r.energy_hartree,residual=r.orbital_residual,iterations=r.iterations,counts=counts,seconds=time.monotonic()-start))
   result_rows.insert(0,json.loads((archive/(caseid+'_retained_target_solver.json')).read_text())['results'][0])
   assert result_rows[1]['converged'] and result_rows[1]['correct_root'],result_rows
   if result_rows[0]['converged'] and result_rows[0]['correct_root']:
    assert abs(result_rows[0]['energy_hartree']-result_rows[1]['energy_hartree'])<1e-8
   row=dict(case=caseid,molecule=name,perturbation_norm=amplitude,results=result_rows)
   path.write_text(json.dumps(row,indent=2)+'\n');rows.append(row);print(json.dumps(row),flush=True)
  certified=optimize_roks(mf,reference,occ,orbital_tolerance=1e-9,certify_index=True)
  index=certified.solver_certificate['curvature']['index']
  wrong=optimize_roks(mf,reference,occ,orbital_tolerance=1e-9,target_saddle_index=index+1,strict=False)
  assert not wrong.converged and wrong.solver_certificate['status']=='saddle_index_mismatch'
  checkpoint=root/(name+'_accepted.npz');certified.save_checkpoint(checkpoint)
  with np.load(checkpoint) as saved:assert json.loads(str(saved['metadata']))['history']
  invalid=surface.rotate(reference,np.random.default_rng(97).normal(size=surface.dimension)*.01)
  try:optimize_roks(mf,reference,occ,restart=checkpoint,reference_coefficients=invalid)
  except ValueError:pass
  else:raise AssertionError('mismatched restart reference accepted')
  replay=optimize_roks(mf,reference,occ,restart=checkpoint,orbital_tolerance=1e-9,target_saddle_index=index)
  assert replay.iterations==0 and replay.converged
  (root/(name+'_curvature.json')).write_text(json.dumps(dict(curvature=certified.solver_certificate['curvature'],wrong_index_rejected=True,restart_iterations=0),indent=2)+'\n')
report=dict(host=socket.getfqdn(),baseline_commit='33ff162',rows=rows,trust_correct_roots=sum(row['results'][1]['correct_root'] and row['results'][1]['converged'] for row in rows))
(root/'solver_validation.json').write_text(json.dumps(report,indent=2)+'\n')
