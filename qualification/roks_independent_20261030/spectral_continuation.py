"""Follow the same CSF from the full Fock to qualified spectral Fock.

The functional is fixed; t is a continuation coordinate, never a fitted
parameter. Only t=1 is a prediction. Wrong-identity cold-start saddles are
retained as diagnostics. Reference and target checkpoints are not overwritten.
"""
from remote_bootstrap import BASE,HOST
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import sys,json,numpy as np,time,traceback
from threadpoolctl import threadpool_limits
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,state_from,write,key,digest,EV
from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_roks_singlet
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_apoc.deltascf import singlet_csf_overlap
from full_response import VerticalReference
from pyscf import lib
lib.num_threads(2)
ROOT=BASE/'results/qualified_spectral_continuation';ROOT.mkdir(exist_ok=True)
case=next(c for c in plan() if c['name']=='uracil')
protocol=dict(host=HOST,source_sha256=digest(__file__),target_model_sha256=digest(BASE/'results/qualified_spectral_response/uracil/model.npz'),
 source_model_sha256=digest(BASE/'results/full_response/uracil/model.npz'),schedule=[.25,.5,.75,1.],minimum_csf_overlap=.7,maximum_orbital_residual=1e-9,
 purpose='numerical state continuation, not parameter fitting; only t=1 reported as target prediction',energy_offset=0.,lambda_exchange=.9)
p=ROOT/'protocol.json'
if p.exists():
 existing=json.loads(p.read_text());protocol['source_sha256']=existing['source_sha256'];assert existing==protocol
else:write(p,protocol)
old,xyz=state_from(case['targets'][0]);s=old.reference_state
calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=old.ground_energy_hartree);native=prepare_gfn2_deltascf(calc,xyz,ground_state=ground)
with np.load(BASE/'results/full_response/uracil/model.npz') as f:source={k:f[k].copy() for k in f.files if k!='request_hash'}
with np.load(BASE/'results/qualified_spectral_response/uracil/model.npz') as f:target={k:f[k].copy() for k in f.files if k!='request_hash'}
assert all(np.max(abs(source[k]-target[k]))==0 for k in ['operators','spin','charge','q0'])
rows=[]
with threadpool_limits(limits=2,user_api='blas'):
 for t in case['targets']:
  startrow=json.loads((BASE/'results/full_response/uracil'/t['id']/'result.json').read_text())
  previous=load_derivative_point(startrow['checkpoint'],request_hash=startrow['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
  path=ROOT/t['id'];path.mkdir(exist_ok=True);last=None;failed=False
  for weight in protocol['schedule']:
   data=dict(source,linear=(1-weight)*source['linear']+weight*target['linear']);mh=key(dict(protocol=digest(p),weight=weight))
   ref=VerticalReference(s,native.kernel.partition,data,mh);ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground,spin_reference=ref)
   success=False
   for index in list(dict.fromkeys([previous.orbital_saddle_index,0,1,2,3])):
    dest=path/f't_{weight:.2f}'/f'index_{index}';dest.mkdir(parents=True,exist_ok=True)
    req=key(dict(model=mh,target=t,index=index));cp=dest/'converged.npz';latest=dest/'latest.npz';start=time.perf_counter()
    try:
     if cp.exists():r=load_derivative_point(cp,request_hash=req,coordinates_angstrom=xyz,require_gradient=False)
     else:
      c=previous.common_coefficients.copy()
      if latest.exists():
       with np.load(latest) as f:assert str(f['request_hash'])==req;c=f['coefficients']
      pm=np.array([(c*o)@c.T for o in previous.occupations]);pt=np.array([(c*o)@c.T for o in previous.triplet_occupations]);fm,em=ctx.fock_and_energy(pm);ft,et=ctx.fock_and_energy(pt)
      seed=replace(previous,common_coefficients=c,fock_mixed=fm,fock_triplet=ft,energy_hartree=2*em-et,converged=False,
       orbital_saddle_index=None,target_orbital_saddle_index=None,spin_reference_fingerprint=ref.fingerprint,gradient_hartree_per_bohr=None)
      def checkpoint(point):
       tmp=dest/'latest.tmp.npz';np.savez_compressed(tmp,request_hash=req,coefficients=point.point);tmp.replace(latest);write(dest/'history.json',list(point.history))
      r=evaluate_gfn2_roks_singlet(calc,xyz,*previous.occupations,ground_state=ground,reference_result=seed,spin_reference=ref,minimum_state_overlap=.7,
       orbital_solver='gmf',saddle_index=index,orbital_tolerance=1e-9,maximum_iterations=100,maximum_step=.15,freeze_and_release=False,checkpoint=checkpoint)
      save_derivative_point(cp,r,request_hash=req,coordinates_angstrom=xyz)
     original_score=singlet_csf_overlap(replace(r,common_coefficients=s.coefficients).to_apoc(),r.to_apoc(),s.overlap)[0]
     step_score=singlet_csf_overlap(replace(previous,spin_reference_fingerprint=r.spin_reference_fingerprint).to_apoc(),r.to_apoc(),s.overlap)[0]
     success=bool(original_score>=.7 and step_score>=.7 and r.orbital_residual<=1e-9 and r.orbital_saddle_index==index)
     row=dict(host=HOST,molecule='uracil',state=t['id'],weight=weight,qualified=success,excitation_ev=r.excitation_energy_hartree*EV,
      tbe_ev=t['tbe_ev'],error_ev=r.excitation_energy_hartree*EV-t['tbe_ev'],original_csf_overlap=original_score,previous_csf_overlap=step_score,index=index,
      orbital_residual=r.orbital_residual,checkpoint=str(cp),request_hash=req,seconds_current_run=time.perf_counter()-start,energy_offset=0.,nuclear_derivatives=False)
     write(dest/'result.json',row);print('CONTINUATION',row,flush=True)
     if success:previous=r;last=row;break
    except Exception as exc:write(dest/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));print('FAILED',t['id'],weight,index,str(exc),flush=True)
   if not success:failed=True;break
  final=last if not failed and last and last['weight']==1. else dict(molecule='uracil',state=t['id'],qualified=False,last_qualified_step=last)
  write(path/'result.json',final);rows.append(final)
write(ROOT/'summary.json',dict(host=HOST,protocol_sha256=digest(p),rows=rows,scope='development state continuation; final functional identical to qualified_spectral_response'))
print('CONTINUATION COMPLETE',flush=True)
