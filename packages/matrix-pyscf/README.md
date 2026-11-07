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

Unrestricted DeltaSCF and common-orbital ROKS are separate APIs. A
broken-symmetry Ms=0 DeltaSCF determinant is not a spin-pure singlet; the ROKS
API constructs a two-determinant singlet CSF. Solvent, nonlocal-correlation
functionals, relativistic response, symmetry constraints and smearing are
rejected rather than evaluated with an incomplete orbital response. ECP-based
APOC export is also rejected. Automatic LINK dispatch for this new DFT backend
is not part of these public orbital APIs.

First remote checks use PySCF 2.14.0: water/STO-3G with HF and PBE0, plus
formaldehyde/6-31G with PBE0. Formaldehyde reaches a stationary point with one
negative orbital mode. Water reaches the same determinant energy as PySCF MOM;
this is a numerical equivalence check, not evidence of better accuracy or speed.
Independent scalar-energy differences validate nuclear gradients. All numerical
certificates and executable drivers are in `qualification/dft_oo_20261102`.

## Common-orbital ROKS and analytic nuclear Hessians

`optimize_roks(excited, ground.mo_coeff, occupations, include_hessian=True)`
optimizes the common-orbital functional `2 E_M - E_T`. It constructs a coherent
two-determinant singlet with S²=0 and exports that CSF to APOC. The mixed and
triplet determinant energies are evaluated at the **same** spatial orbitals.
Their independently relaxed UKS energies are not combined.

Both `optimize_roks` and `optimize_deltascf` now accept `include_hessian=True`.
The returned `hessian_hartree_per_bohr2` is the complete Cartesian matrix,
including orbital relaxation. `response_certificate` records the orbital CP
residual, raw Hessian asymmetry and grid-derivative scope. A full analytic
Hessian request also returns the gradient.

The nuclear response differentiates the complete noncanonical energy-weighted
AO density. ROKS uses one coupled common-orbital CP equation. Moving Becke-grid
weights are differentiated through second order; Gaussian AO derivatives through
third order and LibXC vxc/fxc give the XC chain rule. No displaced geometry is
used in production. The derivation is in [OO_RESPONSE.md](OO_RESPONSE.md).

The initial Hessian implementation supports molecular all-electron UHF and
LDA/GGA UKS, including global and range-separated hybrids. It uses full
integrals and the original Becke partition with standard Becke/Treutler radius
adjustment. Density fitting, meta-GGA Hessians, other partitions, solvent and
relativistic response require separate extensions and are rejected. Results are
not an exact DFT spin-projection formula; spin purity refers to the constructed
CSF, while its multiplet-sum energy remains an approximate functional model.

ROKS now defaults to a residual trust-region solver with the exact Jacobian
adjoint, signed preconditioning, accepted-evaluation reuse and optional
freeze-and-release initialization. Pass `reference_coefficients=target` when the
initial guess differs from the state to retain. `certify_index=True` records the
stationary orbital saddle index; `target_saddle_index=k` rejects a different
index. This is a certificate, not index-following optimization. Checkpoints save
the iteration history and may resume without repeating the ground-state SCF.
See [OO_RESPONSE.md](OO_RESPONSE.md) for the numerical method and validation scope.
