"""Warm continuation to a stationary-reference DFT kernel; no recalibrated energies."""
from remote_bootstrap import HOST,BASE
from full_response import *
from threadpoolctl import threadpool_limits
import stationary_response_calibration as calibration
ROOT=None
WARM_FOLDER=None
def run(case):
 r0,xyz=state_from(case['targets'][0]);s=r0.reference_state
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
 ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r0.ground_energy_hartree);ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground)
 data,request=calibrate(case,s,ctx);ref=VerticalReference(s,ctx.kernel.partition,data,request)
 ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground,spin_reference=ref)
 rows=[]
 for target in case['targets']:
  out=ROOT/case['name']/target['id'];out.mkdir(exist_ok=True)
  if (out/'result.json').exists():rows.append(json.loads((out/'result.json').read_text()));continue
  native_old,_=state_from(target)
  warm_path=BASE/'results'/WARM_FOLDER/case['name']/target['id']/'result.json'
  warm=json.loads(warm_path.read_text())
  old=load_derivative_point(warm['checkpoint'],request_hash=warm['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
  good=False;attempts=[]
  for index in list(dict.fromkeys([old.orbital_saddle_index,0,1,2,3])):
   dest=out/f'index_{index}';dest.mkdir(exist_ok=True);req=key(dict(reference=request,target=target,index=index))
   try:
    cp=dest/'converged.npz';latest=dest/'latest.npz';start=time.perf_counter()
    if cp.exists():r=load_derivative_point(cp,request_hash=req,coordinates_angstrom=xyz,require_gradient=False)
    else:
     c=old.common_coefficients.copy()
     if latest.exists():
      with np.load(latest) as f:assert str(f['request_hash'])==req;c=f['coefficients']
     fm,em=ctx.fock_and_energy(np.array([(c*o)@c.T for o in old.occupations]));ft,et=ctx.fock_and_energy(np.array([(c*o)@c.T for o in old.triplet_occupations]))
     seed=replace(old,common_coefficients=c,fock_mixed=fm,fock_triplet=ft,energy_hartree=2*em-et,converged=False,orbital_saddle_index=None,target_orbital_saddle_index=None,spin_reference_fingerprint=ref.fingerprint,gradient_hartree_per_bohr=None)
     def checkpoint(point):
      np.savez_compressed(dest/'latest.tmp.npz',request_hash=req,coefficients=point.point);(dest/'latest.tmp.npz').replace(latest)
      write(dest/'history.json',list(point.history))
     r=evaluate_gfn2_roks_singlet(calc,xyz,*old.occupations,ground_state=ground,reference_result=seed,spin_reference=ref,minimum_state_overlap=.7,
      orbital_solver='gmf',saddle_index=index,orbital_tolerance=1e-9,maximum_iterations=100,maximum_step=.2,freeze_and_release=False,checkpoint=checkpoint)
     save_derivative_point(cp,r,request_hash=req,coordinates_angstrom=xyz)
    score=singlet_csf_overlap(replace(r,common_coefficients=s.coefficients).to_apoc(),r.to_apoc(),s.overlap)[0]
    good=bool(score>=.7 and r.orbital_residual<=1e-9 and r.orbital_saddle_index==index)
    row=dict(host=HOST,molecule=case['name'],state=target['id'],qualified=good,excitation_ev=r.excitation_energy_hartree*EV,tbe_ev=target['tbe_ev'],error_ev=r.excitation_energy_hartree*EV-target['tbe_ev'],
     csf_overlap=score,orbital_residual=r.orbital_residual,index=index,seconds_current_run=time.perf_counter()-start,scalar_energy_offset=0.,scope='vertical only; no nuclear derivatives',checkpoint=str(cp),request_hash=req)
    attempts.append(row);write(dest/'result.json',row);print('FULL RESPONSE',row,flush=True)
    if good:break
   except Exception as exc:write(dest/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
  if not good:row=dict(molecule=case['name'],state=target['id'],qualified=False,attempts=attempts)
  write(out/'result.json',row);rows.append(row)
 write(ROOT/case['name']/'result.json',dict(host=HOST,records=rows))


if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--molecule',required=True);ap.add_argument('--profile',choices=['stationary_raw','stationary_spectral'],required=True);args=ap.parse_args()
 lib.num_threads(2);ROOT=BASE/'results/stationary_response'/args.profile;ROOT.mkdir(parents=True,exist_ok=True)
 WARM_FOLDER='full_response' if args.profile=='stationary_raw' else 'qualified_spectral_response'
 cases=[c for c in plan() if c['name'] in ['nitroxyl','uracil']]
 for group,names in [('B',['acetaldehyde','acetone','formamide']),('C',['glyoxal','pyrazine','tetrazine']),('D',['acrolein','pyrimidine','nitrosomethane'])]:
  cases += [json.loads((BASE/('prepared_'+group)/n/'case.json').read_text()) for n in names]
 case=next(c for c in cases if c['name']==args.molecule)
 def calibrate(case,s,ctx):
  data,r=calibration.calibrate(case,s,ctx);linear=data['linear_raw' if args.profile=='stationary_raw' else 'linear_spectral']
  selected={k:v for k,v in data.items() if k not in ['linear_raw','linear_spectral']};selected['linear']=linear
  request=key(dict(calibration=r,profile=args.profile));out=ROOT/case['name'];out.mkdir(exist_ok=True)
  file=out/'model.npz'
  if not file.exists():np.savez_compressed(file,request_hash=request,**selected)
  return selected,request
 with threadpool_limits(limits=2,user_api='blas'):run(case)
