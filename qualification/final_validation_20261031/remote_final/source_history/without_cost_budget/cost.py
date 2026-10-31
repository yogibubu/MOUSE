"""Fresh-process matched GFN2-orbital timing; preparation is separately itemized."""
from common import *
import argparse,fcntl,resource
from matrix_gfn2.relative_pair_roks import load_relative_pair_roks,evaluate_relative_pair_roks
from matrix_gfn2.ground_referenced_roks import GroundReferencedROKSResult,ground_referenced_roks_analytic_hessian
from matrix_gfn2.multicsf_frame import RelativeActiveFrame

def setup(name):
 multi={'Furan':'B022','Cyclopentadiene':'B016','Phthalimide':'B024'}
 if name in multi:
  model=MultiCSFModel(W/f'models/{name}/{multi[name]}/model.json');return model,calculator(model.anchor),model.xyz,model.anchor,model.responses,model.grounds,None
 folder=Path('/home/enzo/MOUSE-roks-relative-pair-20261030/vertical')/name/'n_pi';point=load_relative_pair_roks(folder/'stationary.npz',request_hash='relative_vertical_control_v1',require_gradient=False);s=point.components[0].reference_state;xyz=s.coordinates_bohr/ANGSTROM_TO_BOHR;calc=calculator(s);ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s);responses=[]
 for profile in ['full_response','qualified_spectral_response']:
  p=OLD/'results'/profile/name/'model.npz'
  with np.load(p) as a:data={k:a[k].copy() for k in ['operators','charge','spin','linear','q0']}
  responses.append(FrozenNonlocalResponseReference.from_vertical_model(s,ctx.kernel.partition,data,sha(p)))
 grounds=[ground(OUT/f'cost_preparation/{name}/{i}',calc,xyz,s,r) for i,r in enumerate([None]+responses)]
 return None,calc,xyz,s,responses,grounds,point

def tda(calc,xyz,s,responses,grounds):
 roots=[];dims=[]
 for response,g in zip(responses,grounds[1:]):
  ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=response);e,c=eigh(g.fock[0],s.overlap);state=replace(s,coefficients=c,fock=g.fock[0],orbital_energies_hartree=e,total_energy_hartree=g.energy_hartree)
  kernel=ctx.density_response_kernel(state);no=s.electron_count//2;ii=np.repeat(np.arange(no),len(c)-no);aa=np.tile(np.arange(no,len(c)),no);ci=c[:,ii];ca=c[:,aa];columns=[]
  for i,a in zip(ii,aa):
   p=np.outer(c[:,i],c[:,a]);f=kernel(.5*(p+p.T));columns.append(np.einsum('mi,mn,ni->i',ci,f,ca))
  matrix=np.diag(e[aa]-e[ii])+2*np.array(columns).T;assert np.max(abs(matrix-matrix.T))<1e-8;roots.append(np.linalg.eigvalsh(matrix));dims.append(len(matrix))
 return dict(roots_per_parent=[len(r) for r in roots],pair_dimensions=dims,lowest_ev=[float(r[0]*EV) for r in roots],orbitals='own stationary corrected GFN2 orbitals, never DFT orbitals')

def run(name,prop,repeat):
 out=OUT/f'cost/{name}/{prop}/{repeat}';out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 started=time.monotonic();model,calc,xyz,s,responses,grounds,old=setup(name);prep=time.monotonic()-started
 calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Unexpected SCC in prepared evaluation'))
 start=time.monotonic();cpu=time.process_time();extra={}
 if prop=='tda':extra=tda(calc,xyz,s,responses,grounds)
 elif model is not None:
  if prop=='energy':
   objects,point=model.objectives(calc,xyz);frame=RelativeActiveFrame(objects);point=frame.polish(point);e,g,h,*_=frame.evaluate(point);extra=dict(energy_hartree=float(e+grounds[0].energy_hartree-.5*sum(g.energy_hartree for g in grounds[1:])),residual=float(np.max(abs(g))))
  else:
   point=evaluate_multicsf_surface(calc,xyz,model,compute_hessian=prop=='hessian');extra=dict(energy_hartree=point.energy_hartree,residual=point.orbital_residual,hessian=point.hessian_hartree_per_bohr2 is not None)
 else:
  point=evaluate_relative_pair_roks(calc,xyz,responses,reference_result=old,minimum_state_overlap=.7,gradient=prop!='energy');total=GroundReferencedROKSResult(point,tuple(grounds));extra=dict(energy_hartree=total.energy_hartree,residual=point.orbital_residual)
  if prop in ['gradient','hessian']:
   # Differentiate all three retained stationary grounds analytically.
   gg=[]
   for r,g in zip([None]+list(responses),grounds):
    ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=r);c=g.coefficients[0];ff,_=ctx.fock_and_energy(g.spin_densities_ao);q=sum((c.T@f@c)*o[None,:] for f,o in zip(ff,g.occupations));gg.append(ctx.analytic_gradient(g.spin_densities_ao,c@(.5*(q+q.T))@c.T))
   gradient=point.gradient_hartree_per_bohr+gg[0]-.5*(gg[1]+gg[2]);extra['gradient_max']=float(np.max(abs(gradient)))
  if prop=='hessian':
   hh=ground_referenced_roks_analytic_hessian(calc,xyz,responses,reference_result=total).hessian_hartree_per_bohr2;extra['hessian_max']=float(np.max(abs(hh)))
 row=dict(status='complete',molecule=name,property=prop,repeat=repeat,host=HOST,threads=2,natoms=len(s.atomic_numbers),nao=len(s.overlap),rank=model.rank if model is not None else 1,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_loading_and_ground_setup_seconds=prep,total_gradient_included=prop in ['gradient','hessian'],preparation_timing='separately archived; absent stage costs are unknown, never zero',scope='fresh process, prepared converged anchor seeds; not an ab initio unprepared solve; both-parent all-root TDA versus one OO state',**extra)
 write(out/'result.json',row);print(name,prop,repeat,row['wall_seconds'],flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('name');a.add_argument('property',choices=['energy','gradient','hessian','tda']);a.add_argument('repeat',type=int);args=a.parse_args();lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 write(OUT/f'cost_executions/{args.name}/{args.property}/{args.repeat}.json',dict(script_sha256=sha(__file__),common_sha256=sha(Path(__file__).parent/'common.py'),protocol_sha256=sha(OUT/'protocol.json')))
 with threadpool_limits():run(args.name,args.property,args.repeat)
