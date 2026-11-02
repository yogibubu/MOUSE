# Molecular OO-DFT / DeltaSCF adapter

`matrix_pyscf.optimize_deltascf` optimizes real UHF/UKS determinants using
ordinary PySCF ground-state energy, Fock and XC-response routines. Spin
occupations stay fixed. Newton–Krylov steps retain negative orbital curvature;
a residual-norm line search and an immutable determinant-overlap guard control
the search. The solver does not call MOM or TDA and does not require energy to
decrease. It does not guarantee a prescribed saddle index or unique excited root.

```python
import numpy as np
from pyscf import dft
from matrix_pyscf import optimize_deltascf

# ground is an already converged RKS reference at the same geometry.
excited = dft.UKS(ground.mol)
excited.xc = ground.xc
excited.grids.level = ground.grids.level
coefficients = np.array([ground.mo_coeff, ground.mo_coeff])
occupations = np.array([ground.mo_occ / 2, ground.mo_occ / 2])
hole = ground.mol.nelectron // 2 - 1
particle = hole + 1
occupations[0, hole] = 0
occupations[0, particle] = 1
result = optimize_deltascf(
    excited, coefficients, occupations,
    ground_energy_hartree=ground.e_tot,
    include_gradient=True,
    checkpoint="excited.npz",
)
print(result.excitation_energy_hartree, result.spin_squared)
```

Electronic calculations and validation for this project run on Oracle or
MAC-STUDIO. The caller controls PySCF threads, memory, basis and functional.
DFT uses the chosen functional directly; semiempirical lambda and bath parameters
are not imported into this adapter.

Implemented:

- Fixed-occupation, unrestricted saddle search with the analytic orbital Hessian
  action (including the XC kernel), without a dense orbital Hessian in production.
- Analytic nuclear gradients after convergence. Occupied and virtual blocks are
  semicanonicalized for the gradient evaluation; the determinant is unchanged.
  DFT gradients include the quadrature-grid response.
- Atomic accepted-state checkpoints preserving the immutable target orbitals.
  Restart validates molecular geometry, basis, functional, spin and occupations.
  It is a same-geometry restart, not automatic transport between geometries.
- APOC determinant export via `result.to_apoc(excited)`. Cross-geometry matching
  requires actual cross-AO overlaps supplied through the existing APOC API.
- Resumable nonconvergence via `PySCFDeltaSCFConvergenceError.result`.

The current implementation is unrestricted DeltaSCF. A broken-symmetry Ms=0
result is **not a spin-pure singlet**. Multiplet-sum ROKS and its coupled nuclear
response, analytic nuclear Hessians and automatic LINK dispatch are subsequent
work; this adapter does not expose them. Solvent, nonlocal-correlation functionals,
relativistic response, symmetry constraints and smearing are rejected rather than
silently evaluated with an incomplete orbital response. ECP-based APOC export is
also rejected.

First remote checks use PySCF 2.14.0: water/STO-3G with HF and PBE0, plus
formaldehyde/6-31G with PBE0. Formaldehyde reaches a stationary point with one
negative orbital mode. Water reaches the same determinant energy as PySCF MOM;
this is a numerical equivalence check, not evidence of better accuracy or speed.
Independent scalar-energy differences validate nuclear gradients. All numerical
certificates and executable drivers are in `qualification/dft_oo_20261102`.
