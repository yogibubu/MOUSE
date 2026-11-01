import socket
assert socket.getfqdn()=='oracle.sns.it'
import sys
sys.path.insert(0,'/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration/final_validation')
import common
import numpy as np
from determinant_mixed import *

def test_determinant_adjugate_jets_at_rank_deficiency():
    rng=np.random.default_rng(81);n=5;d=2
    for nullity in [0,1,2,3]:
        a=np.diag(np.r_[np.linspace(.8,1.2,n-nullity),np.zeros(nullity)])
        first=.1*rng.normal(size=(d,n,n));mixed=.1*rng.normal(size=(d,n,n));direction=first[1]
        det,adj=determinant_adjugate(Jet(a,first,direction,mixed))
        assert abs(det.v-np.linalg.det(a))<1e-12
        identity=a@adj.v-det.v*np.eye(n);assert np.max(abs(identity))<1e-12
        eps=1e-4
        def literal(x,y):
            m=a+x*first[0]+y*direction+x*y*mixed[0]
            co=np.array([[(-1)**(i+j)*np.linalg.det(np.delete(np.delete(m,i,axis=0),j,axis=1)) for j in range(n)] for i in range(n)])
            return np.linalg.det(m),co.T
        values=[literal(eps,eps),literal(eps,-eps),literal(-eps,eps),literal(-eps,-eps)]
        for index,result in enumerate([det,adj]):
            fd=(values[0][index]-values[1][index]-values[2][index]+values[3][index])/(4*eps**2)
            assert np.max(abs(fd-result.ab[0]))<1e-7
