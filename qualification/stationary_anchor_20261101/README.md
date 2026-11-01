# Stationary-anchor development audit (Oracle, 1 November 2026)

This isolated candidate is not a production change or an accuracy correction.
The four states in two molecules were previously exposed during development.
The protocols, raw checkpoints, interrupted joint-frame attempts and scoring
are retained. No excitation energy label enters the model builders. Lambda
remains 0.9 in exchange, F/Q weights remain 0.5/0.5, and no offset is added.

The embedded frozen-core PBE0 anchor is stationary in the retained valence
space. Its independently evaluated full-AO Fock agrees within 5.3e-13 hartree.
The post-validation metric commutators are 8.99e-9 (Criegee) and 3.14e-10
(Furofuran). Calibration response finite differences agree within 3.7e-9.

The fully shared-orbital candidate uses one averaged energy functional,
not an average of independently optimized parent energies. At identical
orbitals its energy, orbital gradient and orbital Hessian equal the exact
F/Q average to less than 1e-10. All four ground-orthogonal stationary targets
converge with maximum KKT residual below 1.45e-10 and retained character
fidelity at least 0.906. The Criegee pi-pi target required a retained certified
F-parent checkpoint; the unsuccessful initial NTO attempt is preserved.

| State | Reference (eV) | Common candidate (eV) | Error (eV) |
|---|---:|---:|---:|
| Criegee I001 | 2.403 | 2.046656 | -0.356344 |
| Criegee I002 | 3.715 | 4.918425 | +1.203425 |
| Furofuran C001 | 5.489 | 5.603317 | +0.114317 |
| Furofuran C002 | 6.027 | 7.072607 | +1.045607 |

The MAE is 0.679923 eV versus 0.616972 eV for the preceding orthogonal
candidate on precisely these four states. Therefore this candidate does not
resolve transferability and must not replace the production model or be
described as an accuracy improvement in the manuscript. The accompanying
same-model TDA uses embedded-DFT-derived stationary orbitals; it is not the
requested native GFN2-orbital TDA benchmark.

No nuclear response for the new ground-orthogonality constraint has been
derived here. Analytic *orbital* Hessians are not nuclear Hessians. Original
production nuclear derivatives are not automatically valid for this candidate.

The exact native response batch implementation is independently checked at
three densities per molecule and five directions, plus full energy/gradient/
orbital-curvature comparisons. It accelerates the small molecule by about 2x
but does not accelerate Furofuran; it remains isolated research code.

Ten tests (three here and seven preceding transferability tests) passed on
Oracle. Three parameter-free Hamiltonian-projection tests belong to the next
separate experiment. All heavy calculations ran on oracle.sns.it under the
shared compute lock. Bulky quadrature, response-progress and intermediate
model caches remain on Oracle. `remote/REMOTE_INVENTORY.json` records their
paths and SHA256 hashes; selected checkpoints and results are mirrored here.
Timing fields include cache reuse and are not cold end-to-end benchmarks.

Primary machine-readable evidence: `score.json`, remote molecular result and
calibration certificates, raw stationary checkpoints and source versions.
