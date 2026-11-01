"""Independent directional finite-difference check of joint constraint curvature."""
from joint_orthogonal import *
from scipy.linalg import expm

def test_relative_frame_constraint_gradient_and_curvature():
    rng=np.random.default_rng(43);n=4;no=2;left,right=np.triu_indices(n,1)
    occ=np.zeros((2,n));occ[:,:no]=1;occ[0,0]=0;occ[0,no]=1
    bases=[SimpleNamespace(rank=1,no=no,left=left,right=right,occupations=[occ],context=SimpleNamespace(state=SimpleNamespace(overlap=np.eye(n)))) for i in range(2)]
    grounds=[np.eye(n),np.eye(n)];point=[]
    for i in range(2):
        k=rng.normal(size=(n,n));point.append(expm(.2*(k-k.T)))
    for frame in overlap_frames(bases,grounds):
        e,g,h,t,rawmap,v=frame.evaluate(point)
        for unused in range(4):
            direction=rng.normal(size=len(g));direction/=np.linalg.norm(rawmap@direction)
            eps=2e-4
            plus=frame.evaluate(frame.retract(point,eps*direction,rawmap,v))[0]
            minus=frame.evaluate(frame.retract(point,-eps*direction,rawmap,v))[0]
            assert abs((plus-minus)/(2*eps)-g@direction)<1e-7
            assert abs((plus-2*e+minus)/eps**2-direction@h@direction)<1e-7
