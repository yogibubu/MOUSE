"""Independent available-root identity audit and real-molecule orbital-jet checks."""
from pathlib import Path
import socket,os,sys,json,ast,fcntl
assert socket.getfqdn()=='oracle.sns.it'
V=Path('/home/enzo/MOUSE-orbital-performance-20261031');R=Path('/home/enzo/MOUSE-state-following-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030');D=R/'reduced_validation';OUT=V/'pi_pi_repair'
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MOUSE_VALIDATION_ROOT=str(V));sys.path.insert(0,str(V/'source/qualification'));import remote_bootstrap
import numpy as np
from scipy.linalg import expm
from pyscf import scf,gto,lib
from threadpoolctl import threadpool_limits
from matrix_gfn2.deltascf_derivatives import load_derivative_point
sys.path.append('/home/enzo/MOUSE-deltascf-validation-20261029/hybrid_spectrum_20261029');from extent_support import source_molecule
lib.num_threads(2)

def main():
 lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 rows=[]
 for resultfile in sorted((OUT/'optimized').glob('*/*/*/result.json')):
  result=json.loads(resultfile.read_text())
  if result['status']!='stationary_candidate':continue
  name=result['molecule'];sid=result['state'];index=result['profile'];folder=resultfile.parent
  cp=folder/('latest_released.npz' if '_fr_' in folder.name else 'latest.npz')
  with np.load(cp) as a:c=a['coefficients'].copy()
  coeff=np.array(result['ci']);rank=len(coeff)
  gp=D/'prepared'/name/'ground.npz'
  with np.load(gp) as a:xyz=a['coordinates_angstrom'].copy();req=str(a['request_hash'])
  s=load_derivative_point(gp,request_hash=req,coordinates_angstrom=xyz,require_gradient=False).reference_state
  reference=P/'blind/mouse_reference'/name;mol=scf.chkfile.load_mol(str(reference/'backend.chk'));small,t=source_molecule(s);cross=gto.intor_cross('int1e_ovlp',mol,small)@t
  with np.load(reference/'reference.npz') as a:cd=a['coefficients'].copy();od=a['occupations'].copy()
  oi=np.flatnonzero(s.alpha_occupations==1);vi=np.flatnonzero(s.alpha_occupations==0);di=np.flatnonzero(od==2);da=np.flatnonzero(od==0);no=len(oi)
  overlap=s.coefficients.T@cross.T@cd
  with np.load(reference/'backend_tda.npz') as a:amplitudes=a['amplitudes'].reshape(-1,len(di),len(da));labels=a['state_symmetries'].copy();converged=a['converged'].copy()
  case=json.loads((D/'prepared'/name/'case.json').read_text());target=next(t for t in case['targets'] if t['id']==sid);root=target['reference_root'];wanted=labels[root-1]
  h=s.coefficients[:,oi].T@s.overlap@c[:,:rank];p=s.coefficients[:,vi].T@s.overlap@c[:,no:no+rank];transition=(h*coeff)@p.T;transition/=np.linalg.norm(transition)
  candidates=[]
  for i,x in enumerate(amplitudes):
   if labels[i]!=wanted or not converged[i]:continue
   x=x/np.linalg.norm(x);q=overlap[np.ix_(oi,di)]@x@overlap[np.ix_(vi,da)].T;score=float(abs(np.sum(q*transition)))
   candidates.append(dict(root=i+1,projected_overlap=score,projection_norm=float(np.linalg.norm(q))))
  own=next(x for x in candidates if x['root']==root);margin=own['projected_overlap']-max((x['projected_overlap'] for x in candidates if x['root']!=root),default=0.)
  row=dict(molecule=name,state=sid,profile=index,attempt=folder.name,target_root=root,overlap_margin=margin,matched=bool(margin>=.05 and result['fidelity']**2>.5),reference_coverage_complete=False,candidates=candidates)
  rows.append(row)
 # Nonlinear GFN2 + projected DFA kernels: test the actual new objective at a
 # converged molecular point, not just the independent synthetic HF limit.
 import optimize_multicsf as worker
 tree=ast.parse(Path(worker.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='start' for t in n.targets));stop=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration')
 env=dict(vars(worker),name='Furan',sid='B022',index=0,out=OUT/'optimized/Furan/B022/0')
 exec(compile(ast.Module(body=fn.body[start:stop],type_ignores=[]),worker.__file__,'exec'),env)
 obj=env['obj'];c=env['c'];e,g,h=obj.evaluate(c,True);rng=np.random.default_rng(3110);checks=[]
 for k in range(3):
  d=rng.normal(size=len(g));d/=np.linalg.norm(d);rotation=np.zeros_like(c);rotation[obj.left,obj.right]=d;rotation[obj.right,obj.left]=-d;eps=5e-4
  ep=obj.evaluate(c@expm(eps*rotation),False)[0];em=obj.evaluate(c@expm(-eps*rotation),False)[0]
  first=abs((ep-em)/(2*eps)-g@d);second=abs((ep-2*e+em)/eps**2-d@h@d);checks.append(dict(gradient_error=first,hessian_directional_error=second));assert first<2e-7 and second<2e-6,checks[-1]
 report=dict(host=socket.getfqdn(),rows=rows,actual_molecular_orbital_jet_checks=checks,nuclear_derivatives_tested=False,scope='available same-symmetry root audit; not proof of complete spectrum')
 (OUT/'multicsf_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
