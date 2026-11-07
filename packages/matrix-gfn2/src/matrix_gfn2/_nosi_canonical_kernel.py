"""Optional exact integer-mask Slater--Condon loop compiled by Numba.

No fast-math or integral/configuration screening. The Python reference remains
available when Numba is absent or the spin-orbital mask exceeds 64 bits.
"""
import numpy as np
try:
    from numba import njit
except ImportError:
    njit = None

if njit is not None:
    @njit(cache=True)
    def _count(x):
        x=x-((x>>np.uint64(1))&np.uint64(0x5555555555555555))
        x=(x&np.uint64(0x3333333333333333))+((x>>np.uint64(2))&np.uint64(0x3333333333333333))
        x=(x+(x>>np.uint64(4)))&np.uint64(0x0F0F0F0F0F0F0F0F)
        return int((x*np.uint64(0x0101010101010101))>>np.uint64(56))

    @njit(cache=True)
    def _index(x):
        i=0
        while x>np.uint64(1):
            x=x>>np.uint64(1);i+=1
        return i

    @njit(cache=True)
    def _toggle(mask,index,phase):
        bit=np.uint64(1)<<np.uint64(index)
        if _count(mask&(bit-np.uint64(1)))%2:phase=-phase
        return mask^bit,phase

    @njit(cache=True)
    def _matrix(masks,occupied,hs,g):
        nd=len(masks);out=np.zeros((nd,nd))
        for i in range(nd):
            occ=occupied[i];diagonal=0.;two=0.
            for p in occ:
                diagonal+=hs[p,p]
                for q in occ:two+=g[p,q,p,q]
            out[i,i]=diagonal+.5*two
            for j in range(i):
                bra=masks[i];ket=masks[j];diff=bra^ket;count=_count(diff)
                if count>4:continue
                added=bra&diff;removed=ket&diff
                if count==2:
                    p=_index(added);q=_index(removed);value=hs[p,q]
                    for o in occupied[j]:
                        if o!=q:value+=g[p,o,q,o]
                    mask,phase=_toggle(ket,q,1);mask,phase=_toggle(mask,p,phase);value*=phase
                elif count==4:
                    pp=added&(~added+np.uint64(1));p=_index(pp);r=_index(added^pp)
                    qq=removed&(~removed+np.uint64(1));q=_index(qq);s=_index(removed^qq)
                    mask,phase=_toggle(ket,q,1);mask,phase=_toggle(mask,s,phase)
                    mask,phase=_toggle(mask,r,phase);mask,phase=_toggle(mask,p,phase)
                    value=phase*g[p,r,q,s]
                else:raise ValueError('duplicate determinant')
                out[i,j]=value;out[j,i]=value
        return out


def matrix(masks,occupied,hs,g):
    if njit is None or len(hs)>64:return None
    return _matrix(np.asarray(masks,dtype=np.uint64),occupied,hs,g)


if njit is not None:
    @njit(cache=True)
    def _first128(lo,hi):
        if lo:return _index(lo&(~lo+np.uint64(1)))
        return 64+_index(hi&(~hi+np.uint64(1)))

    @njit(cache=True)
    def _toggle128(lo,hi,index,phase):
        if index<64:
            lo,phase=_toggle(lo,index,phase)
        else:
            if _count(lo)%2:phase=-phase
            hi,phase=_toggle(hi,index-64,phase)
        return lo,hi,phase

    @njit(cache=True)
    def _contracted128(masks,occupied,hs,g,starts,columns,weights,ncsf):
        # Contract the sparse determinant-to-CSF transform immediately.
        # No dense determinant Hamiltonian is allocated.
        out=np.zeros((ncsf,ncsf));nd=len(masks)
        for i in range(nd):
            diagonal=0.
            for p in occupied[i]:
                diagonal+=hs[p,p]
                for q in occupied[i]:diagonal+=.5*g[p,q,p,q]
            for a in range(starts[i],starts[i+1]):
                for b in range(starts[i],starts[i+1]):
                    out[columns[a],columns[b]]+=weights[a]*weights[b]*diagonal
            for j in range(i):
                dlo=masks[i,0]^masks[j,0];dhi=masks[i,1]^masks[j,1]
                count=_count(dlo)+_count(dhi)
                if count>4:continue
                alo=masks[i,0]&dlo;ahi=masks[i,1]&dhi
                rlo=masks[j,0]&dlo;rhi=masks[j,1]&dhi
                p=_first128(alo,ahi);q=_first128(rlo,rhi)
                if count==2:
                    value=hs[p,q]
                    for o in occupied[j]:
                        if o!=q:value+=g[p,o,q,o]
                    lo,hi,phase=_toggle128(masks[j,0],masks[j,1],q,1)
                    lo,hi,phase=_toggle128(lo,hi,p,phase);value*=phase
                elif count==4:
                    alo,ahi,_=_toggle128(alo,ahi,p,1);rlo,rhi,_=_toggle128(rlo,rhi,q,1)
                    r=_first128(alo,ahi);s=_first128(rlo,rhi)
                    lo,hi,phase=_toggle128(masks[j,0],masks[j,1],q,1)
                    lo,hi,phase=_toggle128(lo,hi,s,phase)
                    lo,hi,phase=_toggle128(lo,hi,r,phase)
                    lo,hi,phase=_toggle128(lo,hi,p,phase);value=phase*g[p,r,q,s]
                else:raise ValueError('duplicate determinant')
                for a in range(starts[i],starts[i+1]):
                    for b in range(starts[j],starts[j+1]):
                        v=weights[a]*weights[b]*value;ca=columns[a];cb=columns[b]
                        out[ca,cb]+=v;out[cb,ca]+=v
        return out


def contracted_matrix(masks,occupied,hs,g,transform):
    """Exact CSF Hamiltonian with at most 128 spin orbitals and no dense Hd."""
    if njit is None or len(hs)>128:return None
    words=np.array([(m&((1<<64)-1),m>>64) for m in masks],dtype=np.uint64)
    t=transform.tocsr()
    return _contracted128(words,occupied,hs,g,t.indptr,t.indices,t.data,t.shape[1])
