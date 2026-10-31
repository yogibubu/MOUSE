"""Remote research candidate: native ground plus mean corrected ROKS gaps.
All components independently stationary; analytic energy, gradient and Hessian.
"""
import importlib.util,json,time,sys,traceback
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-roks-direct-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from matrix_gfn2.restricted_ground import evaluate_restricted_ground,restricted_ground_analytic_hessian
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.deltascf_analytic_hessian import evaluate_gfn2_deltascf_analytic_hessian
from matrix_gfn2 import combine_roks_results
from matrix_link.optimizer import _exact_cartesian_normal_modes
np=p.np

def write(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def run(name, branch="ground_replacement_newton"):
 data=json.loads((BASE/'results'/name/'prepared.json').read_text());root=BASE/'results'/name/branch;root.mkdir(exist_ok=True)
 if (root/'complete.json').exists():
  previous=json.loads((root/'complete.json').read_text())
  if previous['status']=='stationary':return
  (root/'complete.json').rename(root/'complete_before_adaptive.json')
 refs=[p.load_response_reference(x) for x in data['response_paths']]
 calcs=[p.GFN2Calculator(data['atoms'],temperature_kelvin=0.) for _ in range(3)]
 for calc in calcs:calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('unexpected SCC'))
 seeds=[]
 for profile in ['full_response','qualified_spectral_response']:
  path=Path('/home/enzo/MOUSE-roks-50-50-optimization-20261030/results')/name/profile/'seed.npz'
  with np.load(path) as d:xyz=d['coordinates_angstrom'].copy()
  seeds.append(load_derivative_point(path,request_hash='retained_vertical',coordinates_angstrom=xyz,require_gradient=False))
 state=dict(excited=seeds,ground=[None]*3);anchors=seeds
 def evaluate(x,accepted):
  xyz=x.reshape(-1,3)/p.ANGSTROM_TO_BOHR;ex=[];gr=[]
  for i in range(2):
   r=p.evaluate_gfn2_roks_singlet(calcs[i+1],xyz,*seeds[i].occupations,
    spin_reference=refs[i],reference_result=accepted['excited'][i],reference_policy='geometry_only',
    orbital_solver='newton',certify_index=True,orbital_tolerance=1e-10,maximum_evaluations=150,gradient=True)
   if abs(r.orbital_saddle_index-seeds[i].orbital_saddle_index)>1:
    raise ValueError('stationary branch index outside the predefined adjacent range')
   from matrix_gfn2.roks_tracking import active_orbital_character
   score=active_orbital_character(anchors[i],r,refs[i].anchor_coordinates_bohr)['transition']
   if score<.7:raise ValueError('stationary branch loses initial n-pi pair: '+str(score))
   ex.append(r)
  composite=combine_roks_results(*ex,component_tracking='active_pair')
  for i,ref in enumerate([None]+refs):
   gr.append(evaluate_restricted_ground(calcs[i],xyz,template=seeds[0].reference_state,
    spin_reference=ref,reference_result=accepted['ground'][i],gradient=True))
  energy=gr[0].energy_hartree+.5*sum(a.energy_hartree-b.energy_hartree for a,b in zip(ex,gr[1:]))
  gradient=gr[0].gradient_hartree_per_bohr+.5*sum(a.gradient_hartree_per_bohr-b.gradient_hartree_per_bohr for a,b in zip(ex,gr[1:]))
  return float(energy),gradient.ravel(),dict(excited=ex,ground=gr,composite=composite)
 origin=root if (root/'accepted_geometry.npz').exists() else BASE/'results'/name/'ground_replacement'
 if (origin/'accepted_geometry.npz').exists():
  with np.load(origin/'accepted_geometry.npz') as d:xyz=d['xyz'].copy();h=d['hessian_seed'].copy()
  state['excited']=[load_derivative_point(origin/f'excited_{i}.npz',request_hash='ground_replacement_v1',coordinates_angstrom=xyz,require_gradient=False) for i in range(2)]
  if origin!=root:
   with np.load(origin/'analytic_hessian.npz') as d:total=d['hessian'].copy()
   modes=_exact_cartesian_normal_modes(data['atoms'],xyz,total)
   direction=np.array(modes.cartesian_directions[0]);direction/=np.linalg.norm(direction)
   xyz=xyz+.05*direction.reshape(-1,3);h=np.eye(xyz.size)*.3
 else:h=np.eye(xyz.size)*.3
 anchors=[load_derivative_point(BASE/'results'/name/'ground_replacement'/f'excited_{i}.npz',request_hash='ground_replacement_v1',coordinates_angstrom=np.asarray(json.loads((BASE/'results'/name/'ground_replacement/complete.json').read_text())['xyz']),require_gradient=False) for i in range(2)]
 if branch!='ground_replacement' and not (root/'analytic_preconditioner.json').exists():
  source=BASE/'results'/name/'ground_replacement/analytic_hessian.npz'
  with np.load(source) as d:original=d['hessian'].copy()
  values,vectors=np.linalg.eigh(original);h=(vectors*np.maximum(abs(values),1e-4))@vectors.T
  write(root/'analytic_preconditioner.json',dict(source=str(source),policy='absolute analytic curvature, floor 1e-4; approximate optimization preconditioner only'))
 x=xyz.ravel()*p.ANGSTROM_TO_BOHR
 start=time.monotonic();energy,g,state=evaluate(x,state)
 trace=[];status='max_steps'
 for k in range(121):
  row=dict(step=k,energy=energy,max_gradient=float(np.max(abs(g))),gap_ev=float((energy-state['ground'][0].energy_hartree)*27.211386245988),component_csf=state['composite'].component_csf_overlap,active_pair=state['composite'].component_active_pair_overlap)
  trace.append(row);write(root/'trace.json',trace);print(name,json.dumps(row),flush=True)
  np.savez_compressed(root/'accepted_geometry.npz',xyz=x.reshape(-1,3)/p.ANGSTROM_TO_BOHR,gradient=g,hessian_seed=h)
  for i,r in enumerate(state['excited']):save_derivative_point(root/f'excited_{i}.npz',r,request_hash='ground_replacement_v1',coordinates_angstrom=x.reshape(-1,3)/p.ANGSTROM_TO_BOHR)
  if row['max_gradient']<3e-5:status='stationary';break
  val,vec=np.linalg.eigh(.5*(h+h.T));step=-vec@((vec.T@g)/np.maximum(val,1e-5));norm=np.linalg.norm(step)
  if norm>.15:step*=.15/norm
  accepted=False;errors=[]
  for fraction in [1.,.5,.25,.125,.0625,.03125,.015625,.0078125]:
   dx=fraction*step
   try:
    ee,gg,ss=evaluate(x+dx,state)
    if ee<=energy+1e-4*np.dot(g,dx):accepted=True;break
   except Exception as exc:errors.append(str(exc))
  if not accepted:
   status='line_search_stalled';write(root/'last_rejections.json',errors);break
  y=gg-g;hs=h@dx;sy=np.dot(dx,y);shs=np.dot(dx,hs)
  if shs>1e-16:
   if sy<.2*shs:
    theta=.8*shs/(shs-sy);y=theta*y+(1-theta)*hs;sy=np.dot(dx,y)
   h+=np.outer(y,y)/sy-np.outer(hs,hs)/shs
  else:h=np.eye(x.size)*.3
  x,energy,g,state=x+dx,ee,gg,ss
 out=dict(molecule=name,status=status,steps=k,maximum_gradient=float(np.max(abs(g))),seconds=time.monotonic()-start,energy=energy,xyz=(x.reshape(-1,3)/p.ANGSTROM_TO_BOHR).tolist())
 if status=='stationary':
  xyz=x.reshape(-1,3)/p.ANGSTROM_TO_BOHR
  h0=[restricted_ground_analytic_hessian(c,xyz,reference_result=r,spin_reference=ref).hessian_hartree_per_bohr2 for c,r,ref in zip(calcs,state['ground'],[None]+refs)]
  he=[evaluate_gfn2_deltascf_analytic_hessian(c,xyz,*r.occupations,mode='roks',reference_result=r,spin_reference=ref).hessian_hartree_per_bohr2 for c,r,ref in zip(calcs[1:],state['excited'],refs)]
  total=h0[0]+.5*(he[0]+he[1]-h0[1]-h0[2]);modes=_exact_cartesian_normal_modes(data['atoms'],xyz,total)
  np.savez_compressed(root/'analytic_hessian.npz',hessian=total,frequencies_cm=modes.frequencies_cm,cartesian_directions=modes.cartesian_directions)
  out['frequencies_cm']=list(modes.frequencies_cm)
 write(root/'complete.json',out);print('COMPLETE',json.dumps(out),flush=True)

if __name__=='__main__':
 with p.threadpool_limits(limits=2):run(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'ground_replacement_newton')
