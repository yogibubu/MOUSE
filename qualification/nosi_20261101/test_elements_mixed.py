import socket
assert socket.getfqdn()=='oracle.sns.it'
import sys
sys.path.insert(0,'/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration/final_validation')
import common
import numpy as np
from elements_mixed import Jet,element
from slater_condon import element as literal

def test_singular_safe_element_jets():
    rng=np.random.default_rng(53);n=6;occupied=3;d=2
    h=rng.normal(size=(n,n));h=.5*(h+h.T);v=rng.normal(size=(n,n,n,n));v=.5*(v+v.swapaxes(0,1));v=.5*(v+v.swapaxes(2,3));v=.5*(v+v.transpose(2,3,0,1))
    for nullity in [0,1,2,3]:
        a=np.eye(n)[:,:occupied];b=np.eye(n)[:,list(range(nullity,occupied))+list(range(occupied,occupied+nullity))]
        bases=[a,a,b,b];first=[.1*rng.normal(size=(d,n,occupied)) for _ in bases];second=[.1*rng.normal(size=(d,d,n,occupied)) for _ in bases]
        second=[.5*(x+x.swapaxes(0,1)) for x in second]
        jets=[Jet(x,dx,dx[1],ddx[:,1]) for x,dx,ddx in zip(bases,first,second)]
        analytic=element(jets[:2],jets[2:],h,v)
        def curve(x):
            c=[base+np.einsum('a,aij->ij',x,da)+.5*np.einsum('a,b,abij->ij',x,x,dda) for base,da,dda in zip(bases,first,second)]
            return np.array(literal(c[:2],c[2:],h,v))
        zero=curve(np.zeros(d));assert np.max(abs(zero-[x.v for x in analytic]))<1e-10
        eps=2e-4;eye=np.eye(d)
        for axis in range(d):
            gradient=(curve(eps*eye[axis])-curve(-eps*eye[axis]))/(2*eps)
            assert np.max(abs(gradient-[x.a[axis] for x in analytic]))<1e-7
            if axis==1:mixed=(curve(eps*eye[axis])-2*zero+curve(-eps*eye[axis]))/eps**2
            else:mixed=(curve(eps*(eye[axis]+eye[1]))-curve(eps*(eye[axis]-eye[1]))-curve(eps*(-eye[axis]+eye[1]))+curve(-eps*(eye[axis]+eye[1])))/(4*eps**2)
            assert np.max(abs(mixed-[x.ab[axis] for x in analytic]))<2e-6
