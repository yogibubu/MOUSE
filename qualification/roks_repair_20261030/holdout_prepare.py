"""Prepare the preregistered four-molecule holdout; never change its profile."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,json,time,traceback,numpy as np
from types import SimpleNamespace
from dataclasses import replace
from pyscf import scf,gto,symm,lib
OLD=Path('/home/enzo/MOUSE-deltascf-validation-20261029/hybrid_spectrum_20261029');sys.path.insert(0,str(OLD))
import frozen_relaxation as frozen
from frozen_relaxation import write,digest,key,EV
from extent_support import source_molecule
from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_deltascf,evaluate_gfn2_roks_singlet
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.response_reference import stationary_response_reference
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.roks import ROKSSingletResult
from matrix_gfn2.deltascf_derivatives import save_derivative_point,load_derivative_point
from matrix_apoc.deltascf import singlet_csf_overlap
lib.num_threads(2);OUT=BASE/'holdout_data';OUT.mkdir(exist_ok=True)
PROTOCOL=json.loads((BASE/'results/holdout_protocol.json').read_text())
PARENT=Path('/home/enzo/MATRIX-runs/tda-dft-independent-20261022-v1/docs/reports/independent_dft_tda_20261022')

def prepare(case):
 name=case['name'];out=OUT/name;out.mkdir(exist_ok=True)
 done=out/'case.json'
 if done.exists():return json.loads(done.read_text())
 folder=PARENT/'results'/case['folder'];meta=PARENT/'inputs'/case['metadata']
 selected=json.loads(meta.read_text())[case['metadata_index']]
 assert selected['Spin']==1 and selected['V/R']=='V' and selected['Type']==case['family']
 target=dict(id='n_pi' if case['family']=='npi' else 'pi_pi',tbe_ev=selected['TBE/AVTZ'])
 with np.load(folder/'reference.npz') as f:ref={k:f[k] for k in f.files}
 xyz=ref['coordinates_bohr']/ANGSTROM_TO_BOHR;zs=ref['atomic_numbers']
 request=key(dict(protocol_sha256=digest(BASE/'results/holdout_protocol.json'),reference_sha256=digest(folder/'reference.npz'),
  metadata_sha256=digest(meta),case=case))
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in zs],temperature_kelvin=0.)
 groundfile=out/'ground.npz'
 if groundfile.exists():gs=load_derivative_point(groundfile,request_hash=request,coordinates_angstrom=xyz,require_gradient=False);s=gs.reference_state
 else:
  ground,_=stationary_response_reference(calc,xyz);s=ground.orbital_state
  gs=evaluate_gfn2_deltascf(calc,xyz,s.alpha_occupations,s.beta_occupations,ground_state=ground,
   selection='aufbau',orbital_tolerance=1e-10,density_tolerance=1e-10,energy_tolerance=1e-12,maximum_iterations=500)
  save_derivative_point(groundfile,gs,request_hash=request,coordinates_angstrom=xyz)
 ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=gs.energy_hartree)
 ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground)
 mol=scf.chkfile.load_mol(str(folder/'backend.chk'));mol.symmetry=True;mol.build(False,False)
 assert mol.groupname=='C2v'
 labels=symm.label_orb_symm(mol,mol.irrep_id,mol.symm_orb,ref['coefficients'],s=ref['overlap'],check=False)
 target_irrep=symm.irrep_name2id(mol.groupname,case['symmetry'])
 occupied=np.flatnonzero(s.alpha_occupations+s.beta_occupations==2);virtual=np.flatnonzero(s.alpha_occupations+s.beta_occupations==0)
 di=np.flatnonzero(ref['occupations']==2);da=np.flatnonzero(ref['occupations']==0)
 small,t=source_molecule(s);cross=(gto.intor_cross('int1e_ovlp',mol,small)@t).T
 overlap=s.coefficients.T@cross@ref['coefficients'];oi=overlap[np.ix_(occupied,di)];ov=overlap[np.ix_(virtual,da)]
 with np.load(folder/'backend_tda.npz') as f:vectors=f['amplitudes'].reshape(-1,len(di),len(da));energies=f['energies_hartree']
 irrep_config=np.bitwise_xor(labels[di,None],labels[None,da]);choice=None;roots=[]
 for k,vector in enumerate(vectors):
  assert abs(np.sum(vector**2)-1)<1e-6
  weight=float(np.sum(vector[irrep_config==target_irrep]**2))
  anchor=oi@vector@ov.T;norm=float(np.linalg.norm(anchor))
  roots.append(dict(root=k+1,target_irrep_weight=weight,compact_projection_norm=norm))
  if weight>=.90 and norm>=.7:
   choice=(k,anchor/norm,norm);break
 write(out/'reference_selection.json',dict(host=HOST,case=case,roots=roots,group=mol.groupname,
  protocol_sha256=digest(BASE/'results/holdout_protocol.json'),metadata_sha256=digest(meta),reference_sha256=digest(folder/'reference.npz')))
 if choice is None:raise ValueError('no valence reference state passes the registered symmetry/projection rule')
 k,anchor,norm=choice;pair=np.unravel_index(np.argmax(abs(anchor)),anchor.shape)
 hole=int(occupied[pair[0]]);particle=int(virtual[pair[1]])
 if abs(anchor[pair])<.7:raise ValueError('lowest valence state is not sufficiently described by a single promotion')
 target.update(hole=hole,particle=particle,reference_root=k+1,reference_transition_norm=norm,
  reference_leading_weight=float(anchor[pair]**2),family=case['family'])
 np.savez_compressed(out/'reference_transition.npz',amplitude=anchor,occupied=occupied,virtual=virtual,
  reference_root=k+1,compact_projection_norm=norm)
 occ=np.array([s.alpha_occupations,s.beta_occupations]);occ[0,hole]=0;occ[0,particle]=1
 trip=np.array([np.maximum(*occ),np.minimum(*occ)])
 attempts=[];accepted=None
 for index in PROTOCOL['saddle_indices']:
  state_dir=out/f'index_{index}';state_dir.mkdir(exist_ok=True);state_key=key(dict(request=request,index=index,hole=hole,particle=particle))
  checkpoint=state_dir/'converged.npz';latest=state_dir/'latest.npz'
  try:
   if checkpoint.exists():r=load_derivative_point(checkpoint,request_hash=state_key,coordinates_angstrom=xyz,require_gradient=False)
   else:
    c=s.coefficients.copy()
    if latest.exists():
     with np.load(latest) as f:assert str(f['request_hash'])==state_key;c=f['coefficients'].copy()
    fm,em=ctx.fock_and_energy(np.array([(c*o)@c.T for o in occ]));ft,et=ctx.fock_and_energy(np.array([(c*o)@c.T for o in trip]))
    seed=ROKSSingletResult(c,occ,trip,s.overlap,fm,ft,2*em-et,1.,0,False,
     ground_energy_hartree=gs.energy_hartree,reference_state=s,core_electrons=zs-np.asarray(calc._valence))
    def save(point):
     tmp=latest.with_suffix('.tmp.npz');np.savez_compressed(tmp,coefficients=point.point,request_hash=state_key);tmp.replace(latest)
     write(state_dir/'history.json',list(point.history))
    t0=time.perf_counter()
    r=evaluate_gfn2_roks_singlet(calc,xyz,*occ,ground_state=ground,reference_result=seed,
     minimum_state_overlap=0.,orbital_solver='gmf',saddle_index=index,orbital_tolerance=1e-9,
     maximum_iterations=100,maximum_step=.2,freeze_and_release=True,freeze_iterations=8,checkpoint=save)
    save_derivative_point(checkpoint,r,request_hash=state_key,coordinates_angstrom=xyz)
   csf=singlet_csf_overlap(replace(r,common_coefficients=s.coefficients).to_apoc(),r.to_apoc(),s.overlap)[0]
   h=s.coefficients[:,occupied].T@s.overlap@r.common_coefficients[:,hole]
   p=s.coefficients[:,virtual].T@s.overlap@r.common_coefficients[:,particle]
   rank=np.outer(h,p);transition=float(abs(np.sum(anchor*rank))/np.linalg.norm(rank))
   good=csf>=.7 and transition>=.7 and r.orbital_saddle_index==index and r.orbital_residual<=1e-9
   attempt=dict(index=index,observed_index=r.orbital_saddle_index,csf_overlap=csf,transition_overlap=transition,
    residual=r.orbital_residual,qualified=good)
   attempts.append(attempt);write(state_dir/'summary.json',attempt)
   if good:
    target.update(checkpoint=str(checkpoint),request_hash=state_key,
     source_initialization_seconds=sum(h.get('seconds',0) for h in attempts),
     canonical_csf_overlap=csf,reference_transition_overlap=transition)
    accepted=r;break
  except Exception as exc:
   attempt=dict(index=index,error=str(exc),traceback=traceback.format_exc());attempts.append(attempt);write(state_dir/'summary.json',attempt)
  print('SOURCE ATTEMPT',name,attempts[-1],flush=True)
 write(out/'source_attempts.json',attempts)
 if accepted is None:raise ValueError('no source state passed the registered identity and stationarity rules')
 result=dict(name=name,cohort='new_holdout_20261030',reference=str(folder),targets=[target])
 write(done,result);print('PREPARED',name,target,flush=True);return result

if __name__=='__main__':
 cases=[];excluded=[]
 for case in PROTOCOL['cases']:
  try:cases.append(prepare(case))
  except Exception as exc:
   row=dict(molecule=case['name'],error=str(exc),traceback=traceback.format_exc());excluded.append(row)
   write(OUT/case['name']/'exclusion.json',row);print('EXCLUDED',row,flush=True)
 write(OUT/'plan.json',dict(host=HOST,protocol_sha256=digest(BASE/'results/holdout_protocol.json'),cases=cases,excluded=excluded))
 frozen.OUT=OUT/'frozen_relaxation';frozen.OUT.mkdir(exist_ok=True)
 write(frozen.OUT/'plan.json',dict(host=HOST,cases=cases))
 for case in cases:frozen.run(case)
