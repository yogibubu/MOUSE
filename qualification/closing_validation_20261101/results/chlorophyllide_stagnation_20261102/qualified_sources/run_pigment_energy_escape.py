"""Analytic direct-energy escape from a nonzero residual-norm minimum."""
from pathlib import Path
import socket,json,time,argparse,hashlib
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
from run_pigment_roks import load,HERE,write
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from matrix_gfn2._nosi_orthogonality import ground_overlap_jet,exponential_pullback
from matrix_gfn2._roks_matrix_free import solve
from threadpoolctl import threadpool_limits
p=argparse.ArgumentParser();p.add_argument('--state',type=int,choices=[1,2,3,4],default=2);p.add_argument('--freeze-open',action='store_true');p.add_argument('--seed',choices=['checkpoint','native','state4','orthogonal_native4'],default='checkpoint');args=p.parse_args()
variant='freeze_open_'+args.seed if args.freeze_open else 'energy_escape'
existing=HERE/f'pigment_roks_variants/{variant}/Chlorophyllide_a/state_{args.state}/certificate.json'
if existing.exists():
 record=json.loads(existing.read_text());assert record['report']['converged'];assert record['model_sha256']==hashlib.sha256((HERE/'pigment_models/Chlorophyllide_a/manifest.json').read_bytes()).hexdigest();print('Already certified',existing,flush=True);raise SystemExit(0)
with threadpool_limits(8):
 model,state,reference,calc,base,context=load('Chlorophyllide_a');ground=state.coefficients;no=state.electron_count//2;n=len(ground);s=state.overlap;source=HERE/f'pigment_roks/Chlorophyllide_a/state_{args.state}/checkpoint.npz'
 with np.load(source) as z:c=z['coefficients'].copy();mu=float(z['multiplier']);model_hash=str(z['model_sha256'])
 variant='freeze_open_'+args.seed if args.freeze_open else 'energy_escape'
 if args.seed=='native':
  hole,particle=[(no-1,no),(no-2,no),(no-1,no+1),(no-2,no+1)][args.state-1];order=[hole]+[i for i in range(no) if i!=hole]+[particle]+[i for i in range(no,n) if i!=particle];c=ground[:,order];mu=0.
 if args.seed=='state4':
  with np.load(HERE/'pigment_roks/Chlorophyllide_a/state_4/checkpoint.npz') as z:c=z['coefficients'].copy()
  if args.state in [1,3]:c[:,[0,no-1]]=c[:,[no-1,0]]
  if args.state in [1,2]:c[:,[no,no+1]]=c[:,[no+1,no]]
  mu=0.
 if args.seed=='orthogonal_native4':
  assert args.state==3
  with np.load(HERE/'pigment_roks/Chlorophyllide_a/state_4/checkpoint.npz') as z:known=z['coefficients'].copy()
  native_holes=ground[:,[no-2,no-1]];projection=native_holes.T@s@known[:,0];projection/=np.linalg.norm(projection);other=np.array([-projection[1],projection[0]])
  hole,particle=(no-1,no+1);order=[hole]+[i for i in range(no) if i!=hole]+[particle]+[i for i in range(no,n) if i!=particle];c=ground[:,order];c[:,0]=native_holes@other;c[:,no-1]=native_holes@projection;mu=0.
 assert model_hash==hashlib.sha256((model/'manifest.json').read_bytes()).hexdigest()
 out=HERE/f'pigment_roks_variants/{variant}/Chlorophyllide_a/state_{args.state}';out.mkdir(parents=True,exist_ok=True)
 if (out/'checkpoint.npz').exists():
  with np.load(out/'checkpoint.npz') as z:c=z['coefficients'].copy();mu=float(z['multiplier']);assert str(z['model_sha256'])==model_hash
 occ=np.zeros((2,n));occ[:,:no]=1;occ[0,0]=0;occ[0,no]=1;triplet=np.array([np.maximum(*occ),np.minimum(*occ)]);left,right=np.triu_indices(n,1);keep=np.any(np.vstack([occ,triplet])[:,left]!=np.vstack([occ,triplet])[:,right],axis=0);left,right=left[keep],right[keep]
 optids=np.flatnonzero((left!=0)&(right!=0)&(left!=no)&(right!=no)) if args.freeze_open else np.arange(len(left));optleft,optright=left[optids],right[optids]
 promotions=[(no-1,no),(no-2,no),(no-1,no+1),(no-2,no+1)];transition=np.zeros((4,no,n-no))
 for k,(hh,pp) in enumerate(promotions):transition[k,hh,pp-no]=1.
 pground=ground[:,:no]@ground[:,:no].T;eg=context.fock_and_energy(np.array([pground,pground]))[1]
 history=[];start=time.monotonic()
 def evaluate(coeff):
  energy=0.;g=np.zeros(len(left));parts=[]
  for weight,o in [(2.,occ),(-1.,triplet)]:
   densities=np.array([(coeff*x)@coeff.T for x in o]);f,e=context.fock_and_energy(densities);energy+=weight*e;q=sum((coeff.T@ff@coeff)*x[None,:] for ff,x in zip(f,o));g+=2*weight*(q[left,right]-q[right,left]);parts.append((weight,o,f))
  overlap,j=ground_overlap_jet(ground,coeff,s,occ,left,right)
  return float(energy),g,overlap,j,parts
 def save(coeff,multiplier,row):
  tmp=out/'checkpoint.tmp.npz';np.savez_compressed(tmp,coefficients=coeff,multiplier=multiplier,model_sha256=model_hash,trust_radius=float(row.get('trust_radius',.1)));tmp.replace(out/'checkpoint.npz');write(out/'progress.json',row);print(json.dumps(row),flush=True);history.append(row)
 penalty=10.;initial_overlap=None
 for outer in range(8):
  seed=c.copy();energy,g,ov,j,parts=evaluate(seed);initial_overlap=abs(ov) if initial_overlap is None else initial_overlap
  diagonal=np.zeros(len(left))
  for weight,o,f in parts:
   for occupation,ff in zip(o,f):
    levels=np.diag(seed.T@ff@seed);diagonal+=2*weight*(occupation[right]-occupation[left])*(levels[left]-levels[right])
  scale=np.sqrt(np.maximum(abs(diagonal),.02));cache={};calls=[0]
  def objective(x):
   rotation=np.zeros((n,n));rotation[optleft,optright]=x/scale[optids];rotation[optright,optleft]=-x/scale[optids];u=expm(rotation);coeff=seed@u;e,g,ov,j,unused=evaluate(coeff);effective=mu-penalty*ov;local=g-effective*j;gradient=exponential_pullback(rotation,u,optleft,optright,local[optids])/scale[optids];calls[0]+=1;cache.update(x=x.copy(),coeff=coeff,energy=e,g=g,overlap=ov,j=j,effective=effective,objective=e-mu*ov+.5*penalty*ov*ov)
   return cache['objective'],gradient
  def callback(x):
   if not np.array_equal(x,cache['x']):objective(x)
   tangent=cache['g']-cache['effective']*cache['j'];save(cache['coeff'],cache['effective'],dict(stage='augmented-Lagrangian direct energy',frozen_open_orbitals=args.freeze_open,outer=outer,evaluations=calls[0],energy_hartree=cache['energy'],residual=float(np.max(abs(tangent))),residual_l2=float(np.linalg.norm(tangent)),ground_overlap=cache['overlap'],multiplier=cache['effective'],penalty=penalty,wall_seconds=time.monotonic()-start))
  result=minimize(objective,np.zeros(len(optids)),jac=True,method='L-BFGS-B',callback=callback,options=dict(maxiter=60,maxcor=12,maxls=25,ftol=0.,gtol=1e-8))
  objective(result.x);c=cache['coeff'];mu=cache['effective'];e,g,ov,j,unused=evaluate(c);physical=float(np.max(abs((g-mu*j)[optids])));print('INNER',outer,result.message,physical,ov,flush=True)
  if physical<1e-6 and abs(ov)<1e-7:break
  if outer and abs(ov)>.5*initial_overlap:penalty=min(1000.,penalty*3)
  initial_overlap=abs(ov)
 def callback(coeff,multiplier,row):save(coeff,multiplier,dict(stage='full KKT polishing',**row))
 c,mu,report=solve(context,c,ground,transition,mu,linearization_context=base,krylov='gmres',exact_refinement=True,residual_trust_region=True,retract_constraint=True,moving_frame_jacobian=True,trust_krylov_dimension=0,refine_before_trust=True,maximum_step=.5,initial_trust_radius=.1,tolerance=1e-10,maximum_iterations=60,local_state_following=True,checkpoint=callback)
 record=dict(molecule='Chlorophyllide_a',ordinal=args.state,hole=promotions[args.state-1][0],particle=promotions[args.state-1][1],excitation_ev=(report['energy_hartree']-eg)*27.211386245988,ground_energy_hartree=eg,report=report,spin_squared=0.,spin_definition='coherent shared-orbital singlet; ROKS multiplet energy',DFT_imports=0,new_SCC_calls=0,TDA_solves=0,full_XC_quadrature_energy=True,RI_role='preconditioning only',model_sha256=model_hash,source_checkpoint=str(source),seed=args.seed,frozen_open_initialization=args.freeze_open,energy_escape_history=history)
 write(out/'certificate.json',record);print('CERTIFIED',record['excitation_ev'],report['residual'],flush=True)
