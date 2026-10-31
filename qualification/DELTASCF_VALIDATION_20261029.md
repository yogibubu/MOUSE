# Spin-polarized DeltaSCF and spin-adapted ROKS validation

Implemented analytic nuclear gradients and a fully analytic, relaxed Cartesian
Hessian for the spin-polarized MOUSE GFN2 determinant and the two-open-orbital
ROKS singlet. The analytic Hessian uses native fixed-density second derivatives,
analytic spin/AO-metric terms and an indefinite coupled orbital Schur complement.
It does not evaluate displaced geometries. A separate finite-difference Hessian
of analytic gradients supplies the independent nuclear-derivative check.

All molecular calculations ran on the verified remote host `oracle.sns.it`
(Linux x86_64), with OMP/BLAS/MKL thread limits set to two.
MAC-STUDIO was unavailable via SSH.
The laptop edited code, transferred artifacts and ran only small algebraic tests.
The remote campaign is `/home/enzo/MOUSE-deltascf-validation-20261029`.

## Molecular inputs and model

- H2: H(0,0,0), H(0,0,0.85), coordinates in angstrom.
- Formaldehyde: C(0,0,0), O(1.21,0,0), H(-0.58,0.94,0), H(-0.58,-0.94,0).
- Standard gas-phase GFN2, zero electronic temperature, published shell spin
  constants; no NN overlay, solvent, fixed point charges or imposed symmetry.
- The Ms=0 excited initial occupation promotes the alpha HOMO to the LUMO.
  Each displaced solution is projected from and followed against the central
  state's actual cross-AO metric. These fixed geometries are not excited minima;
  no vibrational frequencies or minimum-stability claims are made.
- Determinant controls: density/orbital residuals 1e-10, energy tolerance 1e-12.
  ROKS orbital stationarity is checked to 1e-10 in the derivative comparisons.

## Analytic gradient versus energy differences

A normalized Cartesian direction was checked with a central energy difference
at a displacement of 1e-4 bohr. Errors are hartree/bohr.

| State and molecule | Absolute directional error |
|---|---:|
| Formaldehyde, unrestricted DeltaSCF | 1.97e-10 |
| H2, spin-adapted ROKS | 2.11e-10 |
| Formaldehyde, spin-adapted ROKS | 2.07e-10 |

The gradient includes the spin dependence of Mulliken shell populations and
the correct variational Pulay multiplier. ROKS uses one shared orthonormality
multiplier, not independent stationary gradients of its auxiliary determinants.
Raw logs: [probe.log](results/deltascf_20261029/probe.log) and
[roks_probe.log](results/deltascf_20261029/roks_probe.log).

## Fully analytic Hessian versus independent numerical Hessian

Errors below are maximum absolute Cartesian matrix differences in
hartree/bohr squared. The numerical reference is a central difference of
relaxed analytic gradients. Refinement is quadratic until the electronic
convergence floor becomes visible.

| Molecule / state | Step 0.001 bohr | Step 0.0005 bohr | Step 0.00025 bohr |
|---|---:|---:|---:|
| H2 / DeltaSCF | 4.04e-6 | 1.01e-6 | 2.52e-7 |
| H2 / ROKS singlet | 4.04e-6 | 1.01e-6 | 2.52e-7 |
| Formaldehyde / DeltaSCF | 1.72e-6 | 4.40e-7 | 1.28e-7 |
| Formaldehyde / ROKS singlet | 1.74e-6 | 4.35e-7 | 1.09e-7 |

The coupled-response equation residual is at most 6.7e-16. Raw analytic
reciprocity errors are at most 6.7e-16 hartree/bohr squared; translational
null-vector residuals are at most 1.2e-15. Rotational covariance was checked
including the nonzero central gradient, rather than incorrectly demanding
rotational zero modes at a nonstationary geometry.

Each displaced derivative point is checkpointed with its converged orbitals,
retained basis/state metadata, geometry and request digest. The refinement
script repeats every completed Hessian request and asserts **zero new point
evaluations**, reusing 13 H2 or 25 formaldehyde points. Central states from the
analytic calculation are reused rather than recalculated.

Evidence: [analytic_probe.log](results/deltascf_20261029/analytic_probe.log),
[derivative_refinement.json](results/deltascf_20261029/derivative_refinement.json)
and the per-point NPZ/request files in the adjacent molecule/refinement folders.

## Spin-pure singlet and BS-to-ROKS continuation

The singlet is a normalized two-determinant CSF with common orthonormal spatial
orbitals and exactly two open orbitals. An independent second-quantized test
applies the spin-raising operator and obtains zero, including cases with paired
closed orbitals. Both physical spin-resolved one-particle densities are equal.

The unrestricted occupied eigenvectors may reorder or rotate without changing
the determinant. Consequently BS-to-ROKS initialization uses corresponding
alpha/beta occupied spaces, not occupied MO index identities. The resulting
restricted singlet is reoptimized; this is not a scalar correction to the BS
energy and is not an exact DFT projection energy.

| Molecule | BS determinant S squared | Singlet CSF S squared | ROKS energy difference from an independent seed / hartree |
|---|---:|---:|---:|
| H2 | 1.000000 | 0 | 6.7e-16 |
| Formaldehyde | 1.008826 | 0 | -3.6e-15 |

The approximate ROKS energy is **2 E_mixed - E_triplet**, with the same common
orbitals in both auxiliary determinants. A spin-pure CSF does not turn that
approximate density functional into an exact electronic Hamiltonian. General
multi-pair spin projection, multiconfigurational states and exact projection of
an arbitrary unrestricted determinant are outside this two-open-orbital method.
APOC follows the CSF through its coherent four-term determinant overlap.

Evidence: [spin_adaptation.json](results/deltascf_20261029/spin_adaptation.json)
and [test_roks.py](test_roks.py).

## Independent ORCA 6.1.1 comparison

ORCA `Native-spGFN2-xTB` unrestricted triplets were run with `EnGrad`, zero
smearing and the standard ORCA SCF infrastructure (`UseXTBMixer false`). MOUSE
uses the corresponding promoted triplet and the same spin-polarized model.
These checks are distinct from the internal finite-difference checks above.

| Molecule | MOUSE minus ORCA energy / hartree | Maximum gradient difference / hartree per bohr |
|---|---:|---:|
| H2 triplet | -3.57e-9 | 7.82e-10 |
| Formaldehyde triplet | +1.75e-9 | 1.91e-8 |

Evidence: [orca_spin_comparison.json](results/deltascf_20261029/orca_spin_comparison.json),
ORCA input/output/engrad files in [orca/](results/deltascf_20261029/orca/).
This verifies implementation consistency of the tested spGFN2 states. It does
not validate singlet excitation energies against experiment or correlated theory.

## Tests, provenance and limits

Sixteen algebraic tests pass locally and on Oracle. They cover spin energy/Fock
consistency, restricted/unrestricted metric response, orbital-rotation-invariant
BS initialization, fermionic spin purity, coherent CSF tracking, restart and
MOUSE-to-APOC export. No molecular SCF jobs were run on the laptop.

[environment.json](results/deltascf_20261029/environment.json) records the host,
Python/NumPy/SciPy versions and implementation hashes. The campaign reused the
already compiled compatible Linux native extension from the existing remote
MATRIX environment; no Mac binary was copied or compiled for validation.

The molecular adapter currently supports fewer than 64 atoms under the stated
gas-phase restrictions. The analytic Hessian is a dense molecular reference
implementation; it rejects a singular orbital response rather than silently
regularizing it. GMF, freeze-and-release, excited-state transition properties
and general multi-open-shell singlets remain outside this implementation.

The final implementation was rechecked against all four saved molecular central
states, without restarting any SCF point. See
[final_saved_reference_check.json](results/deltascf_20261029/final_saved_reference_check.json).

Method references: [ROKS and multiplet-sum formalism, Q-Chem manual](https://manual.q-chem.com/6.1/qchem_manual.pdf),
[excited-state orbital stationarity, Hait and Head-Gordon](https://doi.org/10.1021/acs.jctc.9b01127),
and [ORCA native spGFN2 documentation](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/semiempirical.html).
