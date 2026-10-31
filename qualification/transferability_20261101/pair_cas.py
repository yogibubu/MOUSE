"""Research-only spin-singlet pair state interaction, not a production surface.

Basis: closed h^2, open-shell singlet hp, closed p^2. Diagonal energies,
Brillouin couplings and spin-pair transfer use the same density functional.
This closure is exact for Hartree--Fock CAS(2,2); its general density-functional
transferability must be tested. No bare-ERI coupling or energy fit is inserted.
Only vertical diagnostics are provided here, without derivative claims.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PairCASResult:
    matrix: np.ndarray
    energies: np.ndarray
    coefficients: np.ndarray
    singlet_diagonal: float
    spin_exchange: float
    target_root: int
    target_weight: float
    target_margin: float

def pair_cas_matrix(closed_ground,mixed,triplet,closed_double,fock_ground_hp,fock_double_hp):
    """Singlet matrix in the fixed [h², hp(S=0), p²] phase convention."""
    singlet=2*mixed-triplet;exchange=mixed-triplet
    return np.array([[closed_ground,np.sqrt(2)*fock_ground_hp,exchange],
                     [np.sqrt(2)*fock_ground_hp,singlet,np.sqrt(2)*fock_double_hp],
                     [exchange,np.sqrt(2)*fock_double_hp,closed_double]],dtype=float)

def evaluate_pair_cas(context,coefficients,noccupied,*,hole=None,particle=None):
    c=np.asarray(coefficients);n=c.shape[1];no=int(noccupied)
    hole=no-1 if hole is None else int(hole);particle=no if particle is None else int(particle)
    if not 0<=hole<no<=particle<n:raise ValueError('invalid pair CAS orbitals')
    if not np.allclose(c.T@context.state.overlap@c,np.eye(n),atol=1e-8,rtol=0):raise ValueError('pair CAS requires orthonormal common orbitals')
    ground=np.zeros((2,n));ground[:,:no]=1
    mixed=ground.copy();mixed[0,hole]=0;mixed[0,particle]=1
    triplet=np.array([np.maximum(*mixed),np.minimum(*mixed)])
    double=ground.copy();double[:,hole]=0;double[:,particle]=1
    values=[];focks=[]
    for occ in [ground,mixed,triplet,double]:
        density=np.array([(c*x)@c.T for x in occ]);f,e=context.fock_and_energy(density);values.append(float(e));focks.append(f)
    f0=float(c[:,hole]@(.5*(focks[0][0]+focks[0][1]))@c[:,particle])
    f2=float(c[:,hole]@(.5*(focks[3][0]+focks[3][1]))@c[:,particle])
    matrix=pair_cas_matrix(*values,f0,f2);energies,z=np.linalg.eigh(matrix)
    weights=z[1]**2;root=1+int(np.argmax(weights[1:]));margin=float(weights[root]-np.max(np.delete(weights,root)))
    for a in [matrix,energies,z]:a.setflags(write=False)
    return PairCASResult(matrix,energies,z,2*values[1]-values[2],values[1]-values[2],root,float(weights[root]),margin)
