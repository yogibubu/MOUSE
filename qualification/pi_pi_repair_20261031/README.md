# Minimal-CSF OO-DFT development

This archive develops the pi-to-pi-star correction after closure of the
original 14-state campaign. All reference labels were already exposed. It is
not a new blind benchmark and does not change the public MOUSE default.

The numerical energy is the selected eigenvalue of an orthonormal singlet-CSF
matrix: unchanged spin-polarized ROKS diagonal energies, bare projected
Slater–Condon off-diagonal couplings. Both parents retain weight 1/2, diagonal
exchange-response lambda 0.9, and zero excitation-energy offset. The bare
couplings define a new approximate multistate functional; they are not the
TDA kernel. Spin purity of the CSFs is exact; accuracy of this approximate
energy is a separate issue.

## Results and controls

- The 95%-cumulative-NTO-norm rank trial improves the five pi-pi states to MAE
  0.162 eV, but severely worsens the aza-naphthalene n-pi state. It is rejected
  as a universal prescription; unresolved Rydberg attempts remain archived.
- The inverse-purity effective-rank proposal selects rank two only for
  cyclopentadiene, furan and phthalimide, and reuses identical original
  single-CSF results for the other eleven states. Overall MAE is 0.228 eV;
  pi-pi MAE is 0.214 eV. Matched TDA-Q gives 0.261 and 0.417 eV, respectively.
  This proposal was motivated after examining the previous trial; it needs
  fresh independent validation and rank/anchor sensitivity tests.
- `effective_rank_protocol.json` defines the rule. `select_effective_rank.py`
  reads fixed transition amplitudes only. `score_effective_rank.py` performs
  lightweight, hash-recorded aggregation of the already completed points.
- Nuclear derivative qualification is deliberately separate from these
  vertical energy comparisons. See `THEORY.md`, `STATUS.json`, and reports.

## Implementation

`multicsf_objective.py` contains the energy and exact orbital/CI response.
`multicsf_nuclear_response.py` differentiates the explicitly frozen
co-rotating Lowdin-frame ERI surface and existing GFN2 charge/spin primitives.
`parent_geometry_model.py` reconstructs frozen models from archived inputs.
`multicsf_frame_response.py` preserves the full cross-parent active overlap
matrix with common rotations and independent spectator rotations. Its
analytic curvature includes product-chart and spectator-gauge mixed terms.
The full-frame PES differs from the original individual-cosine PES.

`multicsf_relative_response.py` retains the attempted individual-cosine
extension. Its singular Jacobian has a nonzero second-order obstruction in
furan. Do not bypass that check or discard a dependent constraint.

## Compute and reproduction

All electronic and derivative jobs run on `oracle.sns.it`, checked by each
entry point. They use two threads and the shared campaign compute lock.
The laptop only edits, transfers, aggregates and renders. The original
`MOUSE-orbital-performance-20261031/source` runtime is immutable. The
trusted-Newton core regression used a separate copied runtime and passed
31 tests. The core acceleration is commit `49e0c9d`.

The scripts refer to the archived Oracle campaign paths and its compiled
GFN2/PySCF environment. They are research qualification drivers, not a
portable installation. Run the selected `test_multicsf_*.py` entry point
there only after checking the host. Preserve completed Hessian and orbital
checkpoints. The off-anchor driver caches each completed parent Hessian.

`remote_snapshot/TRANSFER.json` and `pi_pi_export_manifest.json` record
transferred file hashes. Large screened-interaction tensors remain remote;
their hashes are recorded. Earlier blocked paths and setup failures are
retained alongside passing checks, with their original status.
