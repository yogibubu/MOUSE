"""Variational fixed-50/50 ROKS with a shared active pair and independent spectators.

No penalty parameter or energy shift. Closed/virtual spectators relax separately
for the two parent Hamiltonians. The two open spatial orbitals are identical.
The coupled analytic orbital Hessian includes the product-chart BCH terms.
This feasibility probe deliberately exposes NO nuclear derivative API.
"""
import os,socket,sys,json,time,hashlib
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from scipy.linalg import expm
import numpy as np
BASE=Path('/home/enzo/MOUSE-roks-shared-resume-20261030');OLD=Path('/home/enzo/MOUSE-roks-independent-20261030');OUT=Path('/home/enzo/MOUSE-roks-shared-pair-20261030')
assert socket.getfqdn()=='oracle.sns.it';os.environ['MOUSE_VALIDATION_ROOT']=str(BASE)
sys.path.insert(0,str(BASE/'source/qualification'));import remote_bootstrap
for path in (BASE/'source/packages').glob('*/src'):sys.path.insert(0,str(path))
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator,load_response_reference,FrozenNonlocalResponseReference
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.roks_orbital_response import orbital_residual_jacobian
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_gfn2.roks_tracking import active_orbital_character
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR

def load(path,request=None):
 with np.load(path) as d:xyz=d['coordinates_angstrom'].copy();req=str(d['request_hash']) if request is None else request
 return load_derivative_point(path,request_hash=req,coordinates_angstrom=xyz,require_gradient=False)

def root_metric(s,power):
 w,v=np.linalg.eigh(s);return (v*w**power)@v.T

class SharedPairObjective:
 def __init__(self,contexts,occupations):
  self.contexts=contexts;self.occ=np.asarray(occupations);self.triplet=np.array([np.maximum(*self.occ),np.minimum(*self.occ)])
  self.n=len(self.occ[0]);self.s=contexts[0].state.overlap;self.left,self.right=np.triu_indices(self.n,1)
  self.opens=np.flatnonzero(self.occ[0]!=self.occ[1]);assert len(self.opens)==2
  shared=np.isin(self.left,self.opens)|np.isin(self.right,self.opens)
  private=(~shared)&(self.occ.sum(axis=0)[self.left]!=self.occ.sum(axis=0)[self.right])
  self.shared=np.flatnonzero(shared);self.private=np.flatnonzero(private);self.ns=len(self.shared);self.np=len(self.private);self.dimension=self.ns+2*self.np
  self.generators=np.zeros((len(self.left),self.n,self.n))
  for a,(i,j) in enumerate(zip(self.left,self.right)):self.generators[a,i,j]=1.;self.generators[a,j,i]=-1.
  self.cache={}
 def seed(self,components):
  root=root_metric(self.s,.5);inverse=root_metric(self.s,-.5);q=[root@c for c in components];opens=self.opens;others=np.array([i for i in range(self.n) if i not in opens])
  for i in range(2):
   if q[0][:,opens[i]]@q[1][:,opens[i]]<0:q[1][:,opens[i]]*=-1
  mean=q[0][:,opens]+q[1][:,opens];u,_,vt=np.linalg.svd(mean,full_matrices=False);common=u@vt
  result=[]
  for qq in q:
   rest=qq[:,others]-common@(common.T@qq[:,others]);u,_,vt=np.linalg.svd(rest,full_matrices=False)
   qq[:,others]=u@vt;qq[:,opens]=common;result.append(inverse@qq)
  return np.asarray(result)
 def maps(self,point):
  a,b=point;v=a.T@self.s@b
  shared=self.generators[self.shared];hs=[shared,v.T@shared@v]
  maps=[np.zeros((len(self.left),self.dimension)) for _ in range(2)]
  for i in range(2):
   maps[i][:,:self.ns]=hs[i][:,self.left,self.right].T
   maps[i][self.private,self.ns+i*self.np+np.arange(self.np)]=1.
  return maps,hs,v
 def retract(self,point,step):
  a,b=point;_,_,v=self.maps(point);g=np.einsum('a,aij->ij',step[:self.ns],self.generators[self.shared]);common=expm(g)
  pa=expm(np.einsum('a,aij->ij',step[self.ns:self.ns+self.np],self.generators[self.private]));pb=expm(np.einsum('a,aij->ij',step[self.ns+self.np:],self.generators[self.private]))
  return np.asarray([a@common@pa,a@common@v@pb])
 def evaluate(self,point,curvature=True):
  if 'point' not in self.cache or not np.array_equal(point,self.cache['point']):
   values=[]
   for c,ctx in zip(point,self.contexts):
    focks=[];energy=0.;gradient=np.zeros(len(self.left))
    for weight,o in [(2.,self.occ),(-1.,self.triplet)]:
     dm=np.array([(c*x)@c.T for x in o]);f,e=ctx.fock_and_energy(dm);energy+=weight*e;focks.append(f)
     q=sum((c.T@ff@c)*x[None,:] for ff,x in zip(f,o));gradient+=weight*2*(q[self.left,self.right]-q[self.right,self.left])
    values.append(dict(energy=energy,g=gradient,focks=focks))
   maps,hs,v=self.maps(point);g=.5*sum(t.T@row['g'] for t,row in zip(maps,values));self.cache.clear();self.cache.update(point=point.copy(),values=values,maps=maps,hs=hs,v=v,g=g,energy=.5*sum(row['energy'] for row in values))
  if curvature and 'h' not in self.cache:
   total=np.zeros((self.dimension,self.dimension))
   for model,(c,ctx,row,t,hs) in enumerate(zip(point,self.contexts,self.cache['values'],self.cache['maps'],self.cache['hs'])):
    state=SimpleNamespace(**vars(ctx.state));state.coefficients=c;state.alpha_occupations,state.beta_occupations=self.occ
    kernel=ctx.density_response_kernel(state);h=np.zeros((len(self.left),len(self.left)))
    for weight,o,f in [(2.,self.occ,row['focks'][0]),(-1.,self.triplet,row['focks'][1])]:
     h+=weight*orbital_residual_jacobian(c,o,f,kernel,ctx.polarization,self.left,self.right)
    h=.5*(h+h.T);total+=.5*t.T@h@t
    # E(exp(Hs) exp(Hp)): mixed second chart derivative includes 1/2[Hs,Hp].
    gg=np.zeros((self.n,self.n));gg[self.left,self.right]=row['g'];gg[self.right,self.left]=-row['g']
    comm=gg@hs-hs@gg
    correction=.25*comm[:,self.left[self.private],self.right[self.private]]
    columns=self.ns+model*self.np+np.arange(self.np)
    total[np.ix_(np.arange(self.ns),columns)]+=correction
    total[np.ix_(columns,np.arange(self.ns))]+=correction.T
   self.cache['h']=.5*(total+total.T)
  return self.cache['energy'],self.cache['g'],self.cache.get('h')

class SharedHoleObjective(SharedPairObjective):
 def __init__(self,contexts,occupations):
  super().__init__(contexts,occupations)
  self.opens=np.flatnonzero(self.occ[0]<self.occ[1]);assert len(self.opens)==1
  shared=np.isin(self.left,self.opens)|np.isin(self.right,self.opens)
  labels=np.vstack([self.occ,self.triplet]);active=np.any(labels[:,self.left]!=labels[:,self.right],axis=0)
  self.shared=np.flatnonzero(shared);self.private=np.flatnonzero((~shared)&active);self.ns=len(self.shared);self.np=len(self.private);self.dimension=self.ns+2*self.np
 def seed(self,components):
  square=root_metric(self.s,.5);inverse=root_metric(self.s,-.5);q=[square@c for c in components];k=int(self.opens[0]);others=np.array([i for i in range(self.n) if i!=k])
  if q[0][:,k]@q[1][:,k]<0:q[1][:,k]*=-1
  common=q[0][:,k]+q[1][:,k];common/=np.linalg.norm(common);result=[]
  for qq in q:
   rest=qq[:,others]-np.outer(common,common@qq[:,others]);u,_,vt=np.linalg.svd(rest,full_matrices=False);qq[:,others]=u@vt;qq[:,k]=common;result.append(inverse@qq)
  return np.asarray(result)

def run(name):
 root=OUT/'results'/name/'orthogonal_particles';root.mkdir(parents=True,exist_ok=True)
 if (root/'result.json').exists():return
 parents=[];contexts=[];anchors=[];ground=[]
 if name=='formamide':
  data=json.loads((Path('/home/enzo/MOUSE-roks-local-20261030/results')/name/'prepared.json').read_text());manifest=json.loads((Path('/home/enzo/MOUSE-roks-local-20261030/results')/name/'optimization/roks_50_50_accepted/accepted.json').read_text());accepted=Path('/home/enzo/MOUSE-roks-local-20261030/results')/name/'optimization/roks_50_50_accepted'
  for i,row in enumerate(manifest['reference']):parents.append(load(accepted/row['path']))
  for i in range(2):anchors.append(load(Path('/home/enzo/MOUSE-roks-local-20261030/results')/name/f'anchor_{i}.npz'));ground.append(load(accepted/manifest['ground_references'][i+1]['path']).energy_hartree)
  refs=[load_response_reference(p) for p in data['response_paths']]
 else:
  refs=[]
  for profile in ['full_response','qualified_spectral_response']:
   report=json.loads((OLD/'results'/profile/name/'n_pi/result.json').read_text());assert report['qualified'];parent=load(Path(report['checkpoint']),report['request_hash']);parents.append(parent);anchors.append(parent);ground.append(parent.ground_energy_hartree)
   state=parent.reference_state;atoms=[{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in state.atomic_numbers];calc=GFN2Calculator(atoms,temperature_kelvin=0.)
   native=prepare_gfn2_deltascf(calc,state.coordinates_bohr/ANGSTROM_TO_BOHR,orbital_reference=state)
   path=OLD/'results'/profile/name/'model.npz'
   with np.load(path) as d:model={k:d[k].copy() for k in ['operators','charge','spin','q0','linear']}
   refs.append(FrozenNonlocalResponseReference.from_vertical_model(state,native.kernel.partition,model,hashlib.sha256(path.read_bytes()).hexdigest()))
  data=dict(atoms=atoms)
 for parent,ref in zip(parents,refs):
  calc=GFN2Calculator(data['atoms'],temperature_kelvin=0.);calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC invoked'))
  contexts.append(prepare_gfn2_deltascf(calc,parent.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR,orbital_reference=parent.reference_state,spin_reference=ref))
 import ast
 from scipy.linalg import null_space
 source=Path('/home/enzo/MOUSE-roks-direct-20261030/source/qualification/probe_orthogonal_roks.py')
 tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['metric_root','solve']]
 namespace=dict(globals(),null_space=null_space)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
 rows=[]
 for i,(parent,ref,ctx,anchor) in enumerate(zip(parents,refs,contexts,anchors)):
  n=len(parent.overlap);clusters=getattr(ref,'frame_clusters',None)
  if clusters is None:
   from matrix_gfn2.nonlocal_response import TransportedOperators
   transport=TransportedOperators(ctx.state,ref.anchor_coordinates_bohr,np.eye(n)[None])
  else:
   from matrix_gfn2.local_frame_response import LocalTransportedOperators
   transport=LocalTransportedOperators(ctx.state,ref.anchor_coordinates_bohr,np.eye(n)[None],clusters)
  native=anchor.reference_state;occupied=native.alpha_occupations>.5
  qocc=transport.u@root_metric(anchor.overlap,.5)@native.coefficients[:,occupied]
  result,c=namespace['solve'](ctx,parent.common_coefficients,parent.occupations,qocc,root,str(i))
  descriptor=replace(parent,common_coefficients=c,spin_reference_fingerprint=ref.fingerprint,spin_frame_clusters=clusters)
  anchor=replace(anchor,spin_reference_fingerprint=ref.fingerprint,spin_frame_clusters=clusters)
  character=active_orbital_character(anchor,descriptor,ref.anchor_coordinates_bohr)
  row=dict(component=i,gap_ev=(result['energy']-ground[i])*27.211386245988,character=character,**result)
  rows.append(row);print('ORTHOGONAL',json.dumps(row),flush=True)
 out=dict(molecule=name,rows=rows,excitation_ev=.5*sum(r['gap_ev'] for r in rows),qualified=all(r['character']['transition']>=.7 for r in rows),nuclear_derivatives=False,zero_offset=True,weights=[.5,.5])
 (root/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=2):run(sys.argv[1])
