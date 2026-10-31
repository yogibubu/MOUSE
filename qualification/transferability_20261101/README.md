# Transferability development — 1 November 2026

The ground-orthogonal stationary-point prototype is numerically qualified on
six previously exposed rank-one states. It improves accuracy, but does **not**
resolve the method's accuracy or basis-representation transferability. No
production surface, manuscript benchmark or public API is replaced by this
research campaign.

All electronic work ran on verified `oracle.sns.it`, with two BLAS/OpenMP
threads and the shared compute lock. Completed DFT references, calibrated F/Q
responses, stationary grounds and excited checkpoints were reused. No new DFT
reference calculation or excitation-energy fit was needed. The F/Q weights
remain 0.5/0.5, exchange lambda 0.9, excitation offset zero. All existing labels
are exposed development data; this is not a new blind validation.

## Main finding and completed intervention

The retained Criegee pi→pi* singlet has squared determinant overlap 0.435598
and 0.425967 with its own F/Q ground states. Ground orthogonality was not
enforced by the relative-parent active-overlap constraint. In contrast,
ground-response quadratic expansion changes its energy by only 0.00259 eV
on average: higher-order density response is not the dominant cause in this
case.

`orthogonal_singlet.py` provides the exact singlet–closed-shell overlap,
analytic orbital gradient, exponential-coordinate gradient pullback and
analytic orbital Hessian. SVD cofactors and second cofactors remain defined
at singular occupied-overlap matrices; inverse determinants are not used.
This is orbital constraint curvature, **not** the full nuclear Hessian of a
new constrained molecular surface.

`stationary_orthogonal_diagnostic.py` solves the energy/orthogonality KKT
stationarity equations, with residual reduction, finite orbital trust steps,
and explicit overlap guards against a change of state. It searches stationary
points rather than minimizing energy. All six states retain the target
(minimum fidelity 0.995884), with maximum KKT residual 1.262e-9 and maximum
squared ground overlap below 7e-21. The new solver resumes its own checkpoints.

The earlier SLSQP comparison in `remote/orthogonal` is retained as a diagnostic:
it can converge to a different excited state. Those incorrect assignments are
not predictions of the stationary-point candidate and are not used in its
accuracy scores.

## Accuracy on the same geometries and states

| Cohort | State | Reference / eV | Original / eV | Orthogonal stationary / eV |
|---|---|---:|---:|---:|
| Primary development | Criegee n→pi* I001 | 2.403 | 2.112688 | 2.112688 |
| Primary development | Criegee pi→pi* I002 | 3.715 | 1.850801 | 4.850224 |
| Primary development | Benzoxadiazole I004 | 4.606 | 4.491158 | 4.491158 |
| Primary development | Hexatriene I005 | 5.341 | 4.912011 | 4.912011 |
| Furofuran development | C001 | 5.489 | 5.522606 | 5.509973 |
| Furofuran development | C002 | 6.027 | 7.067334 | 7.048379 |

Primary matched-state MAE decreases from **0.674586 to 0.492342 eV** on the
same four states. The primary denominator remains six prescribed states:
Cyclobutadiene and Ammonia remain unrepresented by the unchanged compact
space. Furofuran MAE decreases from **0.536970 to 0.521176 eV**, separately.
No pooled blind-validation score is claimed. `score.json` includes every
primary state and the unchanged representation statuses.

Criegee pi→pi* now overshoots by 1.135224 eV and Furofuran C002 by 1.021379 eV.
Removing ground contamination is therefore necessary for a spectroscopic
state interpretation, but is not a sufficient accuracy correction.

## Other bounded diagnostics

The singlet pair closure (`pair_cas.py`) uses h², open-singlet hp and p²
configurations, diagonal functional energies and functional-derived
couplings. An independent spin-projected FCI test verifies the exact HF
CAS(2,2) limit under orbital rotations. Its general DFT closure is approximate.
The frozen-orbital molecular test does not solve the transferability problem;
it is not integrated into the production functional or used as an offset.
`protocol.json` records the pair diagnostic's original frozen rules.

`nonlinearity_diagnostic.py` isolates higher-order functional terms at the
same retained orbitals. These are decomposition values, not optimized
quadratic-model predictions. Four remote tests pass: HF singlet FCI limit,
local overlap gradient including singular determinants, exponential-chart
pullback, and analytic overlap curvature at singular/regular points.

## Work still required before promotion

1. Balanced treatment of the ground and excited multiconfigurational spaces,
   with orthogonality and state identity imposed together. A single extra
   double configuration is insufficient. Criegee and Furofuran are the first
   fixed development controls; adding many molecules cannot replace this.
2. Carry any accepted multistate model onto the production relative-parent
   manifold. The present diagnostic optimizes the parents separately, which
   is an explicitly different model contract.
3. Derive its complete constrained nuclear response, including the geometry
   dependence of the ground determinant, overlap metric, multipliers and
   constraint curvature. Existing analytic nuclear Hessians cannot simply be
   relabelled as Hessians of this new model.
4. Address the absent virtual/transition space for the rejected cases, then
   freeze the model and test a small genuinely independent set. Existing
   exposed states cannot be reused as blind validation.

All research scripts and the converged checkpoints are retained. Production
code and the previously consolidated paper remain unchanged.

## Follow-up: balanced roots and active-orbital covariance

The subsequent bounded test implements an analytic orbital gradient of the
pair-CAS closure (`pair_cas_response.py`) and optimizes the equal average of
its lowest two singlet eigenvalues at common orbitals. This is distinct from
the F/Q 0.5/0.5 mixture; neither mixture uses fitted weights. Only Criegee I002
and Furofuran C002 are tested, with protocols saved before the calculations.
All reference and ground checkpoints are reused on Oracle.

The state-averaged pair gives 0.733537 eV for Criegee I002 and fails the frozen
open-singlet assignment gate. Furofuran C002 gives 6.993352 eV, retaining the
assignment, but remains 0.966352 eV above its reference. These values do not
replace the previously qualified ground-orthogonal model or its scores.

An internal active-orbital rotation audit reveals a structural defect of this
**new pair-CAS closure**, not a new numerical-Hessian defect of the production
ROKS surface. With the active and inactive projectors unchanged, its full
three-singlet spectrum changes by 4.64–5.41 eV for Criegee and 1.90–1.95 eV
for Furofuran. The HF CAS limit is invariant, but the approximate general
density-functional matrix is not. Optimizing its active orbital gauge can
therefore change energies without changing the physical active space.

`covariant_pair.py` implements a parameter-free SO(2) group projection of the
matrix, transforming each rotated matrix back into the same singlet basis
before averaging. This is a separate functional closure, not an empirical
excitation shift. Tests independently verify preservation of the exact HF
matrix and covariance of a deliberately nonlinear test functional.

For both molecules the 32-node projected matrix agrees with 16 nodes and
with a further active rotation to within 3e-13 eV. This resolves that matrix
covariance defect. Its **unoptimized vertical** gaps at ground NTO orbitals
are 6.308568 eV (Criegee) and 8.997747 eV (Furofuran). They establish neither
spectroscopic improvement nor a useful optimized surface. They must not be
compared as optimized energies with the earlier orbital-optimized results.
No new closure is promoted to production and no accuracy limit is declared
resolved.

Seven remote tests now pass, including the new independent energy-difference
check of the pair-matrix analytic orbital gradient and both covariance tests.
The raw state-average and covariance-audit results are retained under
`remote/state_average`, `remote/rotation_audit`, and `remote/covariant_vertical`.
Assignment qualification and numerical covariance qualification are separate
fields; passing the latter does not imply accurate excitation energies.

The remaining development priority is a physically consistent multistate
functional and anchor construction, rather than additional molecular fitting.
The existing full calibration uses PBE0 first/second derivatives at the DFT
embedding of the native GFN2 density; its F/Q corrected ground can differ from
that density. A future consistently stationary embedded-DFT anchor would be
a distinct candidate and requires its own derivative, accuracy and cost tests.
It has not been calculated or validated in this campaign.
