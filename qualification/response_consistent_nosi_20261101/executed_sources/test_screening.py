import numpy as np
from screening import screened

def test_screening_satisfies_Dyson_and_reduces_positive_interaction():
    rng=np.random.default_rng(909);n=5;no=2;c=np.linalg.qr(rng.normal(size=(n,n)))[0];levels=np.arange(n)*.4
    b=rng.normal(size=(8,n,n));b=.5*(b+b.transpose(0,2,1));v=np.einsum('apq,ars->pqrs',b,b);w,cert=screened(v,c,levels,no)
    polarization=np.zeros((n*n,n*n))
    for i in range(no):
        for a in range(no,n):
            density=.5*(np.outer(c[:,i],c[:,a])+np.outer(c[:,a],c[:,i]));weight=cert['excluded_weights'][i,a-no]
            polarization+=4*weight*np.outer(density.ravel(),density.ravel())/(levels[a]-levels[i])
    vm=v.reshape(n*n,n*n);wm=w.reshape(n*n,n*n)
    np.testing.assert_allclose(vm-wm-vm@polarization@wm,0,atol=1e-10)
    assert np.linalg.eigvalsh(vm-wm).min()>-1e-10
    assert np.linalg.eigvalsh(wm).min()>-1e-10
