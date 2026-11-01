"""Residual trust step in physical orbital-rotation and multiplier coordinates."""
import numpy as np

def residual_trust_step(jacobian,residual,scale,radius):
    """Minimize linearized residual subject to ||scale * step|| <= radius."""
    scale=np.asarray(scale)
    if np.any(scale<=0) or radius<=0:raise ValueError('positive coordinate scales and trust radius required')
    u,s,vt=np.linalg.svd(jacobian/scale[None,:],full_matrices=False)
    rhs=u.T@residual;square=s*s
    def trial(mu):
        return -vt.T@np.divide(s*rhs,square+mu,out=np.zeros_like(rhs),where=square+mu>1e-24)
    z=trial(0.)
    if np.linalg.norm(z)>radius:
        lo=0.;hi=max(1.,float(np.max(square)))
        while np.linalg.norm(trial(hi))>radius:hi*=2
        for unused in range(64):
            mid=.5*(lo+hi)
            if np.linalg.norm(trial(mid))>radius:lo=mid
            else:hi=mid
        z=trial(hi)
    return z/scale
