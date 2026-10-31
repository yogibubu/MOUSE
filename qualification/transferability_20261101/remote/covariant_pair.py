"""SO(2) group projection of the research singlet pair matrix.

This defines a different functional closure with no fitted scale or offset.
Its HF limit and active-rotation covariance must be verified independently.
Nuclear derivatives are not supplied by this vertical prototype.
"""
import numpy as np
from pair_cas import evaluate_pair_cas

def pair_rotation(angle):
    c=np.cos(angle);s=np.sin(angle);t=np.sqrt(2)*c*s
    return np.array([[c*c,t,s*s],[-t,c*c-s*s,t],[s*s,-t,c*c]])

def project_pair_matrix(context,c,no,*,nodes=16):
    matrix=np.zeros((3,3))
    for angle in np.arange(nodes)*np.pi/nodes:
        cc=c.copy();co=np.cos(angle);si=np.sin(angle)
        cc[:,0]=co*c[:,0]-si*c[:,no];cc[:,no]=si*c[:,0]+co*c[:,no]
        h=evaluate_pair_cas(context,cc,no,hole=0,particle=no).matrix
        d=pair_rotation(angle);matrix+=d@h@d.T/nodes
    return .5*(matrix+matrix.T)
