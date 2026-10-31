"""Same-model GFN2 TDA and constrained 50/50 control on fixed G states."""
import os,socket,sys,json,hashlib,time
from pathlib import Path
from dataclasses import replace
BASE=Path('/home/enzo/MOUSE-roks-relative-pair-20261030');OLD=Path('/home/enzo/MOUSE-roks-independent-20261030')
assert socket.getfqdn()=='oracle.sns.it';os.environ['MOUSE_VALIDATION_ROOT']=str(BASE)
sys.path.insert(0,str(BASE/'source/qualification'));import remote_bootstrap
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from matrix_gfn2 import (GFN2Calculator,FrozenNonlocalResponseReference,evaluate_relative_pair_roks,
    save_relative_pair_roks,relative_active_targets)
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.restricted_ground import evaluate_restricted_ground
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR

def load(report):
 path=Path(report['checkpoint'])
 with np.load(path) as d:xyz=d['coordinates_angstrom'].copy()
 return load_derivative_point(path,request_hash=report['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
def run(name):
 out=BASE/'validation_G'/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 case=json.loads((OLD/'prepared'/name/'case.json').read_text());target=next(t for t in case['targets'] if t['id']=='n_pi')
 with np.load(target['transition_file']) as d:amplitude=d['amplitude'].copy();occupied=d['occupied'].copy();virtual=d['virtual'].copy()
 parents=[];refs=[];grounds=[];tda=[];assignment=[]
 for profile in ['full_response','qualified_spectral_response']:
  folder=BASE/'results'/profile/name;report=json.loads((folder/'n_pi/result.json').read_text())
  if not report['qualified']:raise ValueError('unqualified assigned parent '+name+' '+profile)
  parent=load(report);s=parent.reference_state;xyz=s.coordinates_bohr/ANGSTROM_TO_BOHR
  atoms=[{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers];calc=GFN2Calculator(atoms,temperature_kelvin=0.)
  calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC invoked'))
  native=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s);modelpath=folder/'model.npz'
  with np.load(modelpath) as d:model={k:d[k].copy() for k in ['operators','charge','spin','q0','linear']}
  ref=FrozenNonlocalResponseReference.from_vertical_model(s,native.kernel.partition,model,hashlib.sha256(modelpath.read_bytes()).hexdigest())
  parent=replace(parent,spin_reference_fingerprint=ref.fingerprint);parents.append(parent);refs.append(ref)
  ground=evaluate_restricted_ground(calc,xyz,template=s,spin_reference=ref,gradient=False);grounds.append(ground)
  save_derivative_point(out/f'{profile}_ground.npz',ground.reference,request_hash='holdout_G_corrected_S0',coordinates_angstrom=xyz)
  h=s.coefficients[:,occupied].T@s.overlap@parent.common_coefficients[:,target['hole']]
  p=s.coefficients[:,virtual].T@s.overlap@parent.common_coefficients[:,target['particle']];rank=np.outer(h,p)
  norm=float(np.linalg.norm(rank));score=float(abs(np.sum(amplitude*rank))/norm)
  assignment.append(dict(transition_overlap=score,hole_sector_norm=float(np.linalg.norm(h)),particle_sector_norm=float(np.linalg.norm(p)),qualified=bool(score>=.7 and np.linalg.norm(h)>=.7 and np.linalg.norm(p)>=.7)))
  f=ground.reference.fock[0];energies,c=eigh(f,s.overlap)
  state=replace(s,coefficients=c,fock=f,orbital_energies_hartree=energies)
  ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=state,spin_reference=ref);kernel=ctx.density_response_kernel(state)
  occ=np.flatnonzero(s.alpha_occupations>.5);virt=np.flatnonzero(s.alpha_occupations<.5);ii=np.repeat(occ,len(virt));aa=np.tile(virt,len(occ));ci=c[:,ii];ca=c[:,aa];columns=[];start=time.perf_counter()
  for i,a in zip(ii,aa):
   p=np.outer(c[:,i],c[:,a]);field=kernel(.5*(p+p.T));columns.append(np.einsum('mi,mn,ni->i',ci,field,ca))
  matrix=np.diag(energies[aa]-energies[ii])+2*np.asarray(columns).T
  assert np.max(abs(matrix-matrix.T))<1e-8
  values,vectors=np.linalg.eigh(matrix);seconds=time.perf_counter()-start
  projected=(c[:,occ].T@s.overlap@s.coefficients[:,occupied])@amplitude@(s.coefficients[:,virtual].T@s.overlap@c[:,virt]);pnorm=np.linalg.norm(projected);scores=abs(projected.ravel()@vectors)/pnorm;k=int(np.argmax(scores));margin=float(scores[k]-np.max(np.delete(scores,k),initial=0.))
  tda.append(dict(profile=profile,energy_ev=float(values[k]*27.211386245988),overlap=float(scores[k]),margin=margin,projection_norm=float(pnorm),qualified=bool(pnorm>=.7 and scores[k]>=.7 and margin>=.05 and values[k]>0),solver_seconds=seconds))
  np.savez_compressed(out/f'{profile}_tda.npz',matrix=matrix,energies=values,vectors=vectors,coefficients=c)
 joint=evaluate_relative_pair_roks(calc,xyz,refs,initial_components=parents)
 save_relative_pair_roks(out/'relative_joint.npz',joint,request_hash='holdout_G_relative_v1')
 gap=(joint.energy_hartree-.5*sum(g.energy_hartree for g in grounds))*27.211386245988
 raw_gap=.5*sum(p.energy_hartree-p.ground_energy_hartree for p in parents)*27.211386245988
 row=dict(host=socket.getfqdn(),molecule=name,weights=[.5,.5],lambda_exchange=.9,offset=0.,
  excitation_ev=gap,archived_ground_gap_ev=raw_gap,independent_vertical_energy_change_hartree=joint.energy_hartree-.5*sum(p.energy_hartree for p in parents),
  target_squared_overlaps=joint.target_squared_overlaps.tolist(),joint_index=joint.orbital_saddle_index,joint_residual=joint.orbital_residual,
  parent_assignments=assignment,qualified=all(a['qualified'] for a in assignment),tda_gfn2=tda,
  tda_mean_control_ev=.5*sum(r['energy_ev'] for r in tda) if all(r['qualified'] for r in tda) else None,
  tda_orbitals='corrected GFN2 restricted ground orbitals; no DFT orbitals',scope='four prospective molecules; no model or weight fitting after protocol freeze')
 row['tbe_ev']=target['tbe_ev'];row['error_ev']=gap-target['tbe_ev']
 (out/'result.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=2):
  for name in ['cyanoformaldehyde','cyclopropenone','pyridazine','imidazole']:
   try:run(name)
   except Exception as exc:print('UNRESOLVED',name,repr(exc),flush=True)
