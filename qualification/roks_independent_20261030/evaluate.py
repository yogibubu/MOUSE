"""Cold-start no-offset ROKS and TDA of the same corrected functional."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import sys,argparse,json,time,traceback,numpy as np
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest,key,EV,state_from,plan
from matrix_gfn2 import GFN2Calculator,load_response_reference,evaluate_gfn2_roks_singlet
from matrix_gfn2.deltascf import prepare_gfn2_deltascf,run_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_apoc.deltascf import singlet_csf_overlap
from scipy.linalg import eigh
from pyscf import lib
lib.num_threads(4);ROOT=BASE/'results/predictions';ROOT.mkdir(exist_ok=True)
LABELS=['shell_charge_spin','tensor_spin','tensor_charge_spin']

def run(case):
 if not case['targets']:return
 name=case['name'];r0,xyz=state_from(case['targets'][0]);s=r0.reference_state
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
 ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r0.ground_energy_hartree)
 occupied=np.flatnonzero(s.alpha_occupations==1);virtual=np.flatnonzero(s.alpha_occupations==0)
 for label in LABELS:
  model=BASE/'results/models'/name/(label+'.npz')
  if not model.exists():continue
  ref=load_response_reference(model);ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground,spin_reference=ref)
  out=ROOT/name/label;out.mkdir(parents=True,exist_ok=True)
  tdafile=out/'tda.npz';tdajson=out/'tda.json';req=key(dict(reference=ref.fingerprint,source=digest(case['targets'][0]['checkpoint']),version='no-offset-comparison-v1'))
  if tdafile.exists():
   with np.load(tdafile) as f:assert str(f['request_hash'])==req;values=f['energies'];vectors=f['vectors'];c=f['coefficients']
   tda_meta=json.loads(tdajson.read_text())
  else:
   start=time.perf_counter();gp=run_deltascf(s.overlap,np.array([s.coefficients]*2),np.array([s.alpha_occupations,s.beta_occupations]),ctx.fock_and_energy,
    selection='aufbau',density_tolerance=1e-10,orbital_tolerance=1e-10,energy_tolerance=1e-12,maximum_iterations=500)
   ground_seconds=time.perf_counter()-start;energies,c=eigh(gp.fock[0],s.overlap)
   st=replace(s,coefficients=c,fock=gp.fock[0],orbital_energies_hartree=energies,total_energy_hartree=gp.energy_hartree)
   kernel=ctx.density_response_kernel(st);ii=np.repeat(occupied,len(virtual));aa=np.tile(virtual,len(occupied));ci=c[:,ii];ca=c[:,aa]
   start=time.perf_counter();columns=[]
   for i,a in zip(ii,aa):
    p=np.outer(c[:,i],c[:,a]);f=kernel(.5*(p+p.T));columns.append(np.einsum('mi,mn,ni->i',ci,f,ca))
   matrix=np.diag(energies[aa]-energies[ii])+2*np.array(columns).T
   assert np.max(abs(matrix-matrix.T))<1e-8
   values,vectors=np.linalg.eigh(matrix);seconds=time.perf_counter()-start
   np.savez_compressed(tdafile,request_hash=req,energies=values,vectors=vectors,coefficients=c,matrix=matrix,
    ground_coefficients=gp.coefficients,ground_energy=gp.energy_hartree)
   tda_meta=dict(host=HOST,request_hash=req,solver_seconds=seconds,ground_seconds=ground_seconds,
    ground_energy_hartree=gp.energy_hartree,ground_orbital_residual=gp.orbital_residual,
    orbitals='ground orbitals optimized with this same corrected GFN2 functional; no DFT MOs',
    scalar_energy_offset=0.,kernel='singlet Hessian of the same functional')
   write(tdajson,tda_meta)
  rows=[]
  for target in case['targets']:
   state=target['id'];dest=out/state;dest.mkdir(exist_ok=True);final=dest/'result.json'
   if final.exists():rows.append(json.loads(final.read_text()));continue
   old,coords=state_from(target);assert np.max(abs(coords-xyz))<1e-9
   anchor=np.zeros((len(occupied),len(virtual)))
   if 'transition_file' in target:
    with np.load(target['transition_file']) as f:anchor=f['amplitude']
   else:anchor[np.flatnonzero(occupied==target['hole'])[0],np.flatnonzero(virtual==target['particle'])[0]]=1.
   projected=(c[:,occupied].T@s.overlap@s.coefficients[:,occupied])@anchor@(s.coefficients[:,virtual].T@s.overlap@c[:,virtual])
   norm=float(np.linalg.norm(projected));scores=abs(projected.ravel()@vectors)/norm;root=int(np.argmax(scores));margin=float(scores[root]-np.max(np.delete(scores,root),initial=0.))
   tdainfo=dict(energy_ev=float(values[root]*EV),overlap=float(scores[root]),margin=margin,projection_norm=norm,
    qualified=bool(norm>=.7 and scores[root]>=.7 and margin>=.05))
   qualified=False;attempts=[]
   indices=list(dict.fromkeys([target.get('index',old.orbital_saddle_index),0,1,2,3]))
   for index in indices:
    attempt=dest/f'index_{index}';attempt.mkdir(exist_ok=True);request=key(dict(reference=ref.fingerprint,target=target,version='cold-roks-v1',index=index));cp=attempt/'converged.npz';latest=attempt/'latest.npz'
    try:
     start=time.perf_counter()
     if cp.exists():r=load_derivative_point(cp,request_hash=request,coordinates_angstrom=xyz,require_gradient=False)
     else:
      coeff=s.coefficients.copy()
      if latest.exists():
       with np.load(latest) as f:assert str(f['request_hash'])==request;coeff=f['coefficients']
      fm,em=ctx.fock_and_energy(np.array([(coeff*o)@coeff.T for o in old.occupations]));ft,et=ctx.fock_and_energy(np.array([(coeff*o)@coeff.T for o in old.triplet_occupations]))
      seed=replace(old,common_coefficients=coeff,fock_mixed=fm,fock_triplet=ft,energy_hartree=2*em-et,
       converged=False,orbital_saddle_index=None,target_orbital_saddle_index=None,spin_reference_fingerprint=ref.fingerprint,gradient_hartree_per_bohr=None)
      def checkpoint(point):
       np.savez_compressed(attempt/'latest.tmp.npz',request_hash=request,coefficients=point.point);(attempt/'latest.tmp.npz').replace(latest)
       write(attempt/'history.json',list(point.history))
      r=evaluate_gfn2_roks_singlet(calc,xyz,*old.occupations,ground_state=ground,reference_result=seed,spin_reference=ref,
       minimum_state_overlap=0.,orbital_solver='gmf',saddle_index=index,orbital_tolerance=1e-9,maximum_iterations=120,
       maximum_step=.2,freeze_and_release=True,freeze_iterations=8,checkpoint=checkpoint)
      save_derivative_point(cp,r,request_hash=request,coordinates_angstrom=xyz)
     csf=singlet_csf_overlap(replace(r,common_coefficients=s.coefficients).to_apoc(),r.to_apoc(),s.overlap)[0]
     h=s.coefficients[:,occupied].T@s.overlap@r.common_coefficients[:,target['hole']]
     p=s.coefficients[:,virtual].T@s.overlap@r.common_coefficients[:,target['particle']];rank=np.outer(h,p)
     transition=float(abs(np.sum(anchor*rank))/np.linalg.norm(rank))
     qualified=bool(csf>=.7 and transition>=.7 and r.orbital_saddle_index==index and r.orbital_residual<=1e-9)
     row=dict(host=HOST,molecule=name,state=state,model=label,qualified=qualified,csf_overlap=csf,transition_overlap=transition,
      orbital_residual=r.orbital_residual,index=index,excitation_ev=r.excitation_energy_hartree*EV,tbe_ev=target['tbe_ev'],
      error_ev=r.excitation_energy_hartree*EV-target['tbe_ev'],seconds_current_run=time.perf_counter()-start,
      ground_energy_hartree=r.ground_energy_hartree,tda=tdainfo,request_hash=request,checkpoint=str(cp),
      model_sha256=digest(model),model_fingerprint=ref.fingerprint,scalar_energy_offset=0.)
     assert abs(r.ground_energy_hartree-tda_meta['ground_energy_hartree'])<1e-9
     write(attempt/'validation.json',row);attempts.append(row);print('ROKS',name,label,state,index,row,flush=True)
     if qualified:break
    except Exception as exc:attempts.append(dict(index=index,error=str(exc),traceback=traceback.format_exc()));write(attempt/'failure.json',attempts[-1]);print('ATTEMPT FAILED',name,label,state,index,str(exc),flush=True)
   if not qualified:row=dict(host=HOST,molecule=name,state=state,model=label,qualified=False,tbe_ev=target['tbe_ev'],tda=tdainfo,attempts=attempts)
   write(final,row);rows.append(row)
  write(out/'result.json',dict(host=HOST,model=label,records=rows))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--group',type=int,default=0);p.add_argument('--groups',type=int,default=1);p.add_argument('--development',action='store_true');a=p.parse_args()
 cases=[c for c in plan() if c['name'] in ['nitroxyl','uracil']] if a.development else [json.loads(p.read_text()) for p in sorted((BASE/'prepared').glob('*/case.json'))]
 for i,c in enumerate(cases):
  if i%a.groups!=a.group:continue
  try:run(c)
  except Exception as exc:write(ROOT/c['name']/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
