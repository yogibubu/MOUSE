"""Canonical state identity and native ROKS controls for the frozen new cohort."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import sys,json,time,traceback,numpy as np,argparse
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest,key,EV
from extent_support import source_molecule
from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_deltascf,evaluate_gfn2_roks_singlet
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.response_reference import stationary_response_reference
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.roks import ROKSSingletResult
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_apoc.deltascf import singlet_csf_overlap
from pyscf import scf,gto,symm,lib
lib.num_threads(4);PROTOCOL=json.loads((BASE/'qualified_spectral_holdout_C_protocol.json').read_text());ROOT=BASE/'prepared_C';ROOT.mkdir(exist_ok=True)

def prepare(case):
 name=case['name'];out=ROOT/name;out.mkdir(exist_ok=True)
 if (out/'case.json').exists():
  previous=json.loads((out/'case.json').read_text())
  if not any("A''" in e['error'] for e in previous['excluded']):return
  (out/'case.json').rename(out/'case_before_prime_normalization.json')
 folder=Path(case['reference']);request=key(dict(protocol=digest(BASE/'qualified_spectral_holdout_C_protocol.json'),reference=digest(folder/'reference.npz')))
 with np.load(folder/'reference.npz') as f:ref={k:f[k] for k in f.files}
 xyz=ref['coordinates_bohr']/ANGSTROM_TO_BOHR
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in ref['atomic_numbers']],temperature_kelvin=0.)
 gp=out/'ground.npz'
 if gp.exists():gs=load_derivative_point(gp,request_hash=request,coordinates_angstrom=xyz,require_gradient=False)
 else:
  ground,_=stationary_response_reference(calc,xyz);s=ground.orbital_state
  gs=evaluate_gfn2_deltascf(calc,xyz,s.alpha_occupations,s.beta_occupations,ground_state=ground,selection='aufbau',
   orbital_tolerance=1e-10,density_tolerance=1e-10,energy_tolerance=1e-12,maximum_iterations=500)
  save_derivative_point(gp,gs,request_hash=request,coordinates_angstrom=xyz)
 s=gs.reference_state;ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=gs.energy_hartree)
 ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground)
 mol=scf.chkfile.load_mol(str(folder/'backend.chk'));mol.symmetry=True;mol.build(False,False)
 labels=symm.label_orb_symm(mol,mol.irrep_id,mol.symm_orb,ref['coefficients'],s=ref['overlap'],check=False)
 di=np.flatnonzero(ref['occupations']==2);da=np.flatnonzero(ref['occupations']==0)
 occupied=np.flatnonzero(s.alpha_occupations==1);virtual=np.flatnonzero(s.alpha_occupations==0)
 small,t=source_molecule(s);cross=(gto.intor_cross('int1e_ovlp',mol,small)@t).T
 overlap=s.coefficients.T@cross@ref['coefficients'];oi=overlap[np.ix_(occupied,di)];ov=overlap[np.ix_(virtual,da)]
 vectorfile=next((folder/p for p in ['baseline.npz','selected_full.npz','backend_tda.npz'] if (folder/p).exists()))
 with np.load(vectorfile) as f:vectors=f['amplitudes'].reshape(-1,len(di),len(da))
 accepted=[];excluded=[]
 for target in case['targets']:
  target=dict(target);state=target['id'];dest=out/state;dest.mkdir(exist_ok=True)
  try:
   if target['prior_assignment_present'] and target['reference_root'] is None:raise ValueError('archived family assignment unresolved')
   irrep=symm.irrep_name2id(mol.groupname,target['symmetry'].replace('_','').replace("''",'"'))
   irrepconfigs=np.bitwise_xor(labels[di,None],labels[None,da]);chosen=None;diagnostics=[]
   roots=[target['reference_root']-1] if target['reference_root'] else range(len(vectors))
   for k in roots:
    v=vectors[k];weight=float(np.sum(v[irrepconfigs==irrep]**2));a=oi@v@ov.T;norm=float(np.linalg.norm(a))
    diagnostics.append(dict(root=k+1,irrep_weight=weight,projection_norm=norm))
    if weight>=.9 and norm>=.7:chosen=(k,a/norm,norm);break
   write(dest/'selection.json',dict(vector_file=str(vectorfile),vector_sha256=digest(vectorfile),roots=diagnostics))
   if chosen is None:raise ValueError('fixed reference root fails symmetry or compact projection')
   k,anchor,norm=chosen;pair=np.unravel_index(np.argmax(abs(anchor)),anchor.shape)
   if abs(anchor[pair])<.7:raise ValueError('reference is not dominated by one compact orbital promotion')
   h=int(occupied[pair[0]]);p=int(virtual[pair[1]])
   np.savez_compressed(dest/'transition.npz',amplitude=anchor,occupied=occupied,virtual=virtual)
   occ=np.array([s.alpha_occupations,s.beta_occupations]);occ[0,h]=0;occ[0,p]=1
   trip=np.array([np.maximum(*occ),np.minimum(*occ)]);good=False
   for index in PROTOCOL['saddle_indices']:
    attempt=dest/f'index_{index}';attempt.mkdir(exist_ok=True);req=key(dict(request=request,state=state,hole=h,particle=p,index=index))
    try:
     cp=attempt/'converged.npz';latest=attempt/'latest.npz';start=time.perf_counter()
     if cp.exists():r=load_derivative_point(cp,request_hash=req,coordinates_angstrom=xyz,require_gradient=False)
     else:
      c=s.coefficients.copy()
      if latest.exists():
       with np.load(latest) as f:assert str(f['request_hash'])==req;c=f['coefficients'].copy()
      fm,em=ctx.fock_and_energy(np.array([(c*o)@c.T for o in occ]));ft,et=ctx.fock_and_energy(np.array([(c*o)@c.T for o in trip]))
      seed=ROKSSingletResult(c,occ,trip,s.overlap,fm,ft,2*em-et,1.,0,False,
       ground_energy_hartree=gs.energy_hartree,reference_state=s,core_electrons=s.atomic_numbers-np.asarray(calc._valence))
      def checkpoint(point):
       np.savez_compressed(attempt/'latest.tmp.npz',request_hash=req,coefficients=point.point);(attempt/'latest.tmp.npz').replace(latest)
       write(attempt/'history.json',list(point.history))
      r=evaluate_gfn2_roks_singlet(calc,xyz,*occ,ground_state=ground,reference_result=seed,minimum_state_overlap=0.,
       orbital_solver='gmf',saddle_index=index,orbital_tolerance=1e-9,maximum_iterations=120,maximum_step=.2,
       freeze_and_release=True,freeze_iterations=8,checkpoint=checkpoint)
      save_derivative_point(cp,r,request_hash=req,coordinates_angstrom=xyz)
     csf=singlet_csf_overlap(replace(r,common_coefficients=s.coefficients).to_apoc(),r.to_apoc(),s.overlap)[0]
     hole=s.coefficients[:,occupied].T@s.overlap@r.common_coefficients[:,h]
     particle=s.coefficients[:,virtual].T@s.overlap@r.common_coefficients[:,p];rank=np.outer(hole,particle)
     tr=float(abs(np.sum(anchor*rank))/np.linalg.norm(rank));good=bool(csf>=.7 and tr>=.7 and r.orbital_saddle_index==index and r.orbital_residual<=1e-9)
     row=dict(qualified=good,index=index,csf_overlap=csf,transition_overlap=tr,residual=r.orbital_residual,excitation_ev=r.excitation_energy_hartree*EV,seconds_current_run=time.perf_counter()-start)
     write(attempt/'validation.json',row);print(name,state,index,row,flush=True)
     if good:
      target.update(hole=h,particle=p,checkpoint=str(cp),request_hash=req,reference_root=k+1,reference_transition_norm=norm,
       native_roks_ev=r.excitation_energy_hartree*EV,index=index,transition_file=str(dest/'transition.npz'))
      accepted.append(target);break
    except Exception as exc:write(attempt/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
   if not good:raise ValueError('no stationary native seed passes fixed identity rules')
  except Exception as exc:excluded.append(dict(state=state,error=str(exc)));print('EXCLUDED',name,state,str(exc),flush=True)
 write(out/'case.json',dict(case,targets=accepted,excluded=excluded,ground_checkpoint=str(gp),ground_request_hash=request,protocol_sha256=digest(BASE/'qualified_spectral_holdout_C_protocol.json')))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--group',type=int,default=0);p.add_argument('--groups',type=int,default=1);a=p.parse_args()
 for i,c in enumerate(PROTOCOL['cases']):
  if i%a.groups!=a.group:continue
  try:prepare(c)
  except Exception as exc:write(ROOT/c['name']/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
