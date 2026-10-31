"""Checkpointed complete analytic Phthalimide Hessian, with stage profiling."""
from common import *
import fcntl,cProfile,pstats,resource
import matrix_gfn2.multicsf_surface as surface
root=OUT/'hessian_acceleration/Phthalimide';root.mkdir(parents=True,exist_ok=True)
lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
source=Path(surface.__file__).parent
signature={name:sha(source/name) for name in ['nonlocal_response.py','multicsf_surface.py','multicsf_nuclear.py','deltascf_analytic_hessian.py']}
write(root/'execution.json',dict(host=HOST,threads=2,source_sha256=signature,driver_sha256=sha(__file__)))
parent_original=surface.nuclear_response;ground_original=surface.restricted_ground_analytic_hessian
model=MultiCSFModel(W/'models/Phthalimide/B024/model.json');calc=calculator(model.anchor)
calc.evaluate=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected SCC'))
def parent(obj,c,eri):

 # Density/model hashes identify each parent independently of call ordering.
 key=hashlib.sha256(obj.context.charge_correction.k.tobytes()+obj.context.charge_correction.v.tobytes()+c.tobytes()).hexdigest()
 path=root/('parent_'+key);path.mkdir(exist_ok=True);meta=path/'source.json'
 if meta.exists() and json.loads(meta.read_text())==signature and (path/'blocks.npz').exists():
  with np.load(path/'blocks.npz') as a:result={k:a[k].copy() for k in a.files}
  print('RESTORED parent',key,flush=True);return result
 start=time.monotonic();result=parent_original(obj,c,eri);np.savez_compressed(path/'blocks.npz',**result);write(meta,signature);print('PARENT',key,time.monotonic()-start,flush=True);return result
def ground(*args,**kwargs):
 start=time.monotonic();result=ground_original(*args,**kwargs);print('GROUND',kwargs.get('spin_reference').fingerprint if kwargs.get('spin_reference') is not None else 'native',time.monotonic()-start,flush=True);return result
surface.nuclear_response=parent;surface.restricted_ground_analytic_hessian=ground
with threadpool_limits():
 start=time.monotonic();cpu=time.process_time();prof=cProfile.Profile();prof.enable()
 try:
  result=surface.evaluate_multicsf_surface(calc,model.xyz,model,compute_hessian=True)
 finally:
  prof.disable();prof.dump_stats(str(root/'profile.pstats'))
  with (root/'profile.txt').open('w') as f:pstats.Stats(prof,stream=f).sort_stats('cumulative').print_stats(45)
 destination=root/'accepted'
 if not destination.exists():save_multicsf_result(destination,result,model)
 hh=result.hessian_hartree_per_bohr2
 write(root/'result.json',dict(status='complete',host=HOST,threads=2,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,energy_hartree=result.energy_hartree,gradient_max=float(np.max(abs(result.gradient_hartree_per_bohr))),residual=result.orbital_residual,hessian_shape=hh.shape,hessian_symmetry_error=float(np.max(abs(hh-hh.T))),model_fingerprint=model.fingerprint,source_sha256=signature,profiled=True))
 print('COMPLETE HESSIAN',time.monotonic()-start,flush=True)
