"""Uniform active-space convergence of constrained-screened spin-pure OO-CID.

No energy-dependent damping, molecular switches or fitted shifts. Complete
rank-two singlet configurations for m highest occupied and m lowest virtual
canonical native orbitals; retained optimized targets/doubles remain unchanged.
"""
from pathlib import Path
import sys,json,socket
assert socket.getfqdn()=='oracle.sns.it'
ROOT=Path('/home/enzo/MOUSE-response-consistent-nosi-20261101');HERE=Path('/home/enzo/MOUSE-pi-pi-transfer-repair-20261101')
sys.path.insert(0,str(ROOT))
from basis import *
from nosi_spectrum import spectrum
from screening import sector
from pauli import normal_ordered_one_body
from scipy.linalg import solve
from itertools import combinations,combinations_with_replacement

def run(item,m):
 cohort,name=item['cohort'],item['molecule'];out=HERE/f'expanded_cid_{m}'/cohort/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 d=prepare_basis(cohort,name);case=d['case'];ctx=d['ctx'];no=d['no'];c=d['canonical'];n=len(c);nr=len(d['rows']);y=d['sr']@c;basis=d['basis'][:1+2*nr];labels=d['labels'][:len(basis)];s=case['s']
 base_count=len(basis);occ0=np.zeros((2,n));occ0[:,:no]=1
 def energy(occ):return float(ctx.fock_and_energy(np.array([(c*x)@c.T for x in occ]))[1])
 eg=float(ctx.fock_and_energy(np.array([d['cg'][:,:no]@d['cg'][:,:no].T]*2))[1]);ham_bare=Hamiltonian(d['h'],d['eri']);diagonal=[0.]
 # Retained target and paired-double functionals exactly match the old basis.
 for row in d['rows']:
  with np.load(d['source']/'coherent_states'/row['id']/'final/final.npz') as z:cc=z['coefficients'].copy()
  mix=occ0.copy();mix[0,0]=0;mix[0,no]=1;trip=np.array([np.maximum(*mix),np.minimum(*mix)])
  def ener(o):return float(ctx.fock_and_energy(np.array([(cc*x)@cc.T for x in o]))[1])
  diagonal.append(2*ener(mix)-ener(trip)-eg)
 for row in d['rows']:
  with np.load(d['source']/'coherent_states'/row['id']/'final/final.npz') as z:cc=z['coefficients'].copy()
  o=occ0.copy();o[:,0]=0;o[:,no]=1;diagonal.append(float(ctx.fock_and_energy(np.array([(cc*x)@cc.T for x in o]))[1])-eg)
 holes=list(range(max(0,no-m),no));particles=list(range(no,min(n,no+m)));four_blocks=[];records=[]
 operators=.5*(s.overlap@np.load(d['source']/'coherent_common/model.npz')['operators']+np.load(d['source']/'coherent_common/model.npz')['operators']@s.overlap)
 spin=np.load(d['source']/'coherent_common/model.npz')['spin'].copy()
 for hi,hj in combinations_with_replacement(holes,2):
  for a,b in combinations_with_replacement(particles,2):
   core=[i for i in range(no) if i not in [hi,hj]]
   if hi==hj and a==b:
    ids=core+[a];csf=closed(y,ids);maximum=max(abs(ham_bare.csf(old,csf)[0]) for old in basis[:base_count])
    if maximum>=.7:continue
    o=np.zeros((2,n));o[:,ids]=1;basis.append(csf);labels.append(f'pair_h{hi}_p{a}');diagonal.append(energy(o)-eg);records.append(dict(holes=[hi,hi],particles=[a,a]))
   elif hi==hj or a==b:
    if hi==hj:paired=core;opened=[a,b]
    else:paired=core+[a];opened=[hi,hj]
    ca=y[:,paired+[opened[0]]];cb=y[:,paired+[opened[1]]];csf=[(2**-.5,(ca,cb)),(2**-.5,(cb,ca))]
    mixed=np.zeros((2,n));mixed[:,paired]=1;mixed[0,opened[0]]=1;mixed[1,opened[1]]=1;triplet=np.zeros((2,n));triplet[:,paired]=1;triplet[0,opened]=1
    basis.append(csf);labels.append(f'open_h{hi}_{hj}_p{a}_{b}');diagonal.append(2*energy(mixed)-energy(triplet)-eg);records.append(dict(holes=[hi,hj],particles=[a,b]))
   else:
    sites=[hi,hj,a,b];order=core+sites+[i for i in range(n) if i not in core+sites];four=csfs(y[:,order],no);start=len(basis);basis.extend(four);labels.extend([f'four_h{hi}_{hj}_p{a}_{b}_0',f'four_h{hi}_{hj}_p{a}_{b}_1'])
    balanced=np.zeros((2,n));balanced[:,core]=1;balanced[:,sites]=.5;ec=energy(balanced);features=np.einsum('aij,ik,jk->ka',operators,c[:,sites],c[:,sites]);block=spin_matrix(ec-eg,features@spin@features.T);diagonal.extend(np.diag(block));four_blocks.append((start,block));records.extend([dict(holes=[hi,hj],particles=[a,b])]*2)
 # Constrained RPA excludes exactly the entire selected canonical active product.
 op=out/'operators.npz'
 if op.exists():
  with np.load(op) as z:w=z['eri'].copy();h=z['h'].copy()
 else:
  co=c[:,:no];cv=c[:,no:];hs=[np.eye(no)[:,i] for i in holes];ps=[np.eye(n-no)[:,a-no] for a in particles]
  for row in d['rows']:
   with np.load(d['source']/'coherent_states'/row['id']/'final/final.npz') as z:cc=z['coefficients'].copy()
   hs.append(co.T@s.overlap@cc[:,0]);ps.append(cv.T@s.overlap@cc[:,no])
  oh=sector(hs,no);ov=sector(ps,n-no);wh=np.sum(oh**2,axis=1);wp=np.sum(ov**2,axis=1);ex=np.clip(1-wh[:,None]*wp[None,:],0,1)
  ii=np.repeat(np.arange(no),n-no);aa=np.tile(np.arange(no,n),no);fg,_=ctx.fock_and_energy(np.array([co@co.T]*2));levels=np.diag(c.T@fg[0]@c);gap=levels[aa]-levels[ii];valid=ex.ravel()>1e-10;ii=ii[valid];aa=aa[valid];gap=gap[valid]
  transition=np.einsum('pi,qi->ipq',y[:,ii],y[:,aa]);transition+=transition.transpose(0,2,1).copy();t=transition.reshape(len(transition),-1).T*np.sqrt(ex.ravel()[valid])[None,:];v=d['eri'].reshape(n*n,n*n);vt=v@t;w=v-vt@solve(np.diag(gap)+t.T@vt,vt.T,assume_a='pos');w=.5*(w+w.T);w=w.reshape((n,)*4);pg=2*y[:,:no]@y[:,:no].T;h=normal_ordered_one_body(w,d['si']@fg[0]@d['si'],pg);np.savez_compressed(op,h=h,eri=w)
 ham=Hamiltonian(h,w);matrix=np.zeros((len(basis),)*2);metric=np.zeros_like(matrix)
 for i in range(len(basis)):
  for j in range(i+1):ov,value=ham.csf(basis[i],basis[j]);metric[i,j]=metric[j,i]=ov;matrix[i,j]=matrix[j,i]=value
 matrix-=matrix[0,0]*metric;residual=np.array(diagonal)/np.diag(metric)-np.diag(matrix)/np.diag(metric);matrix+=.5*metric*(residual[:,None]+residual[None,:])
 for start,block in four_blocks:matrix[start:start+2,start:start+2]=block
 old=d['old'];rows=json.loads((old/'result.json').read_text())['predictions']
 with np.load(old/'checkpoint.npz') as z:ref=np.column_stack([z['coefficients'][:,r['root']] for r in rows])
 pad=np.zeros((len(basis),len(rows)));pad[:len(ref)]=ref;primary,e,v=spectrum(matrix,metric,pad,rows,True)
 np.savez_compressed(out/'checkpoint.npz',matrix=matrix,metric=metric,energies=e,coefficients=v)
 write(out/'result.json',dict(primary=primary,labels=labels,active_holes=holes,active_particles=particles,added_records=records,weights=[.5,.5],lambda_exchange=.9,offset_ev=0));print(name,m,len(basis),primary['predictions'],flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--molecule');parser.add_argument('--active',type=int,default=3);args=parser.parse_args()
 with threadpool_limits():
  for item in json.loads((HERE/'cases.json').read_text()):
   if args.molecule is None or args.molecule==item['molecule']:run(item,args.active)
