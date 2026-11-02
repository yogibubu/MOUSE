"""Two exact four-spin singlets and quadratic-spin-functional completion."""
from itertools import combinations
import numpy as np
ASSIGNMENTS=tuple(combinations(range(4),2))

def spin_vectors():
    vectors=np.zeros((6,2));phase=np.ones(6)
    for row,alpha in enumerate(ASSIGNMENTS):
        spins=[1 if i in alpha else -1 for i in range(4)]
        phase[row]=(-1)**sum(spins[i]<spins[j] for i in range(4) for j in range(i+1,4))
        if spins[0]!=spins[1] and spins[2]!=spins[3]:vectors[row,0]=.5*spins[0]*spins[2]
        if spins[0]==spins[1]:vectors[row,1]=1/np.sqrt(3)
        else:vectors[row,1]=-1/(2*np.sqrt(3))
    return vectors,phase

def csfs(c,no):
    sites=[no-2,no-1,no,no+1];core=list(range(no-2));v,phase=spin_vectors();result=[]
    for column in range(2):
        basis=[]
        for row,alpha in enumerate(ASSIGNMENTS):
            value=v[row,column]*phase[row]
            if not value:continue
            ai=core+[sites[i] for i in alpha];bi=core+[sites[i] for i in range(4) if i not in alpha]
            basis.append((float(value),(c[:,ai],c[:,bi])))
        result.append(basis)
    return result

def matrix(charge_energy,spin_pair):
    h=np.zeros((6,6));self_energy=.5*np.trace(spin_pair)
    for row,alpha in enumerate(ASSIGNMENTS):
        spins=np.array([1 if i in alpha else -1 for i in range(4)])
        h[row,row]=charge_energy+self_energy+sum(spin_pair[i,j]*spins[i]*spins[j] for i in range(4) for j in range(i+1,4))
        for i in range(4):
            for j in range(i+1,4):
                if spins[i]==spins[j]:continue
                changed=set(alpha);changed.symmetric_difference_update([i,j]);column=ASSIGNMENTS.index(tuple(sorted(changed)))
                h[row,column]+=2*spin_pair[i,j]
    vectors,_=spin_vectors();return vectors.T@h@vectors
