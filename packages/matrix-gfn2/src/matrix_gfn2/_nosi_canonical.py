"""Exact canonical-CSF Slater--Condon block without occupied-overlap SVDs."""
import numpy as np
from scipy.sparse import csc_matrix

def _apply(mask,annihilate,create):
 phase=1
 for i in annihilate:
  if not mask>>i&1:return 0
  if (mask&((1<<i)-1)).bit_count()%2:phase=-phase
  mask^=1<<i
 for i in create:
  if mask>>i&1:return 0
  if (mask&((1<<i)-1)).bit_count()%2:phase=-phase
  mask|=1<<i
 return phase

def block(basis,canonical,h,eri,core,active,*,records=None,compiled=True):
 y=np.asarray(canonical);core=list(core);active=list(active);na=len(active);ns=2*na
 ya=y[:,active];pc=2*y[:,core]@y[:,core].T
 if core:
  j=np.einsum('pqrs,rs->pq',eri,pc,optimize=True);k=np.einsum('prsq,rs->pq',eri,pc,optimize=True)
  ec=float(np.sum(pc*h)+.5*np.sum(pc*j)-.25*np.sum(pc*k));hc=ya.T@(h+j-.5*k)@ya
 else:ec=0.;hc=ya.T@h@ya
 # Only active indices need a four-index transformation. The core has
 # already been contracted exactly into its scalar and one-body fields.
 wa=np.einsum('pqrs,pi,qj,rk,sl->ijkl',eri,ya,ya,ya,ya,optimize=True);orb=np.tile(np.arange(na),2);spin=np.repeat(np.arange(2),na)
 hs=hc[np.ix_(orb,orb)]*(spin[:,None]==spin[None,:]);v=wa[np.ix_(orb,orb,orb,orb)].transpose(0,2,1,3)
 mask=(spin[:,None,None,None]==spin[None,None,:,None])&(spin[None,:,None,None]==spin[None,None,None,:]);g=v*mask;g=g-g.swapaxes(2,3)
 rows={};terms=[];mapping={a:i for i,a in enumerate(active)}
 if records is not None:
  from ._nosi_spin import spin_vectors,ASSIGNMENTS
  vectors,phases=spin_vectors()
  for record in records:
   closed=record['closed'];opened=record.get('open',[]);kind=record['kind']
   if kind=='closed':patterns=[(1.,closed,closed)]
   elif kind=='open':patterns=[(2**-.5,closed+[opened[0]],closed+[opened[1]]),(2**-.5,closed+[opened[1]],closed+[opened[0]])]
   else:
    patterns=[]
    for row,alpha in enumerate(ASSIGNMENTS):
     weight=float(vectors[row,record['component']]*phases[row])
     if weight:patterns.append((weight,closed+[opened[j] for j in alpha],closed+[opened[j] for j in range(4) if j not in alpha]))
   expansion=[]
   for weight,alpha,beta in patterns:
    # Shared closed-column permutation occurs twice and cancels.
    key=0;phase=(-1)**sum(c>o for c in closed for o in opened)
    for sp,indices in enumerate((alpha,beta)):
     if len(set(indices))!=len(indices) or not set(core).issubset(indices) or set(indices)-set(core)-set(active):raise ValueError('record outside canonical active space')
     tail=indices[len(closed):];inversions=sum(a>b for j,a in enumerate(tail) for b in tail[j+1:])
     phase*=(-1)**inversions
     for i in indices:
      if i in mapping:key|=1<<(mapping[i]+sp*na)
    if key not in rows:rows[key]=len(rows)
    expansion.append((rows[key],weight*phase))
   terms.append(expansion)
 else:
  for csf in basis:
   expansion=[]
   for weight,pair in csf:
    key=0;phase=1.
    for sp,c in enumerate(pair):
     projection=y.T@c;indices=np.argmax(abs(projection),axis=0);ordered=sorted(indices.tolist())
     if len(set(ordered))!=len(ordered) or not set(core).issubset(ordered) or set(ordered)-set(core)-set(active):raise ValueError('determinant outside canonical active space')
     minor=projection[ordered];factor=float(np.linalg.det(minor))
     if abs(abs(factor)-1)>1e-8 or np.max(abs(y[:,ordered]@minor-c))>1e-8:raise ValueError('noncanonical determinant in fast block')
     phase*=factor
     for i in ordered:
      if i in mapping:key|=1<<(mapping[i]+sp*na)
    if key not in rows:rows[key]=len(rows)
    expansion.append((rows[key],weight*phase))
   terms.append(expansion)
 masks=list(rows);nd=len(masks);ti=[];tj=[];tv=[]
 for j,expansion in enumerate(terms):
  for i,value in expansion:ti.append(i);tj.append(j);tv.append(value)
 transform=csc_matrix((tv,(ti,tj)),shape=(nd,len(terms)))
 occupied={k:np.array([i for i in range(ns) if k>>i&1],dtype=int) for k in masks};hd=None
 if compiled and ns<=64:
  from ._nosi_canonical_kernel import matrix as compiled_matrix
  hd=compiled_matrix(masks,np.array([occupied[k] for k in masks]),hs,g)
 accelerated=hd is not None
 if hd is None:
  hd=np.zeros((nd,nd))
  for i,bra in enumerate(masks):
   occ=occupied[bra];hd[i,i]=hs[occ,occ].sum()+.5*g[occ[:,None],occ[None,:],occ[:,None],occ[None,:]].sum()
   for j in range(i):
    ket=masks[j];diff=bra^ket;count=diff.bit_count()
    if count>4:continue
    added=bra&diff;removed=ket&diff
    if count==2:
     p=added.bit_length()-1;q=removed.bit_length()-1;common=occupied[ket];common=common[common!=q];value=_apply(ket,[q],[p])*(hs[p,q]+g[p,common,q,common].sum())
    elif count==4:
     pp=added&-added;p=pp.bit_length()-1;r=(added^pp).bit_length()-1;qq=removed&-removed;q=qq.bit_length()-1;s=(removed^qq).bit_length()-1
     value=_apply(ket,[q,s],[r,p])*g[p,r,q,s]
    else:raise ValueError('duplicate determinant')
    hd[i,j]=hd[j,i]=value
 metric=(transform.T@transform).toarray();matrix=(transform.T@hd)@transform+ec*metric
 return metric,matrix,dict(compiled_integer_kernel=accelerated,determinants=nd,csfs=len(terms),active_orbitals=na,closed_core_orbitals=len(core))
