import numpy as np
from scipy.sparse import csc_matrix
from matrix_gfn2._nosi_canonical import _apply
from matrix_gfn2._nosi_canonical_kernel import contracted_matrix


def test_exact_contracted_masks_cross_the_64_bit_boundary():
    rng=np.random.default_rng(918);n=68;sites=[1,2,63,64,65,67];h=np.zeros((n,n));g=np.zeros((n,n,n,n))
    a=rng.normal(size=(6,6));h[np.ix_(sites,sites)]=(a+a.T)/2
    v=rng.normal(size=(6,6,6,6));v=v-v.swapaxes(0,1);v=v-v.swapaxes(2,3);v=(v+v.transpose(2,3,0,1))/2
    g[np.ix_(sites,sites,sites,sites)]=v
    occupied=np.array([[1,63,64],[2,63,64],[1,63,67],[1,64,67],[2,65,67]])
    masks=[sum(1<<int(p) for p in row) for row in occupied]
    hd=np.zeros((len(masks),len(masks)))
    for i,bra in enumerate(masks):
        occ=occupied[i];hd[i,i]=h[occ,occ].sum()+.5*g[occ[:,None],occ[None,:],occ[:,None],occ[None,:]].sum()
        for j in range(i):
            ket=masks[j];added=[p for p in range(n) if (bra&~ket)>>p&1];removed=[p for p in range(n) if (ket&~bra)>>p&1]
            if len(added)==1:
                p=added[0];q=removed[0];common=occupied[j][occupied[j]!=q]
                value=_apply(ket,[q],[p])*(h[p,q]+g[p,common,q,common].sum())
            elif len(added)==2:
                p,r=added;q,s=removed;value=_apply(ket,[q,s],[r,p])*g[p,r,q,s]
            else:value=0.
            hd[i,j]=hd[j,i]=value
    t=csc_matrix(np.array([[1.,0.,0.],[.2,.8,0.],[0.,.6,0.],[0.,0.,1.],[0.,.3,.4]]))
    actual=contracted_matrix(masks,occupied,h,g,t)
    assert actual is not None
    np.testing.assert_allclose(actual,t.T@hd@t,atol=1e-12)
