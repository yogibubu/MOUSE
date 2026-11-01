"""Exact canonical-CSF Slater--Condon block without occupied-overlap SVDs."""
import numpy as np

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

def block(basis,canonical,h,eri,core,active):
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
 masks=list(rows);nd=len(masks);transform=np.zeros((nd,len(basis)))
 for j,expansion in enumerate(terms):
  for i,value in expansion:transform[i,j]+=value
 occupied={k:np.array([i for i in range(ns) if k>>i&1],dtype=int) for k in masks};hd=np.zeros((nd,nd))
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
 metric=transform.T@transform;matrix=transform.T@hd@transform+ec*metric
 return metric,matrix,dict(determinants=nd,csfs=len(basis),active_orbitals=na,closed_core_orbitals=len(core))
