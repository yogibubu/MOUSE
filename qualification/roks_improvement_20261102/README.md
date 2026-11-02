ROKS numerical improvement, 2 November 2026
==========================================

All electronic jobs ran on verified MAC-STUDIO, never on the laptop.
`results/operators.json` checks the exact nonstationary moving-frame Jacobian
adjoint, a finite-difference Jacobian action, the trust ball and lazy response.
`results/final_solver/solver_validation.json` contains the latest-source six
retained-target recovery checks. Baseline entries are reused completed results
from the exact committed sources in `baseline_sources/`, not rerun timings.
The new results recover six correct targets versus three for commit 33ff162.
These are two molecules and three perturbed guesses each, not six novel states.
Both algorithms retain the same known stationary target. Automatic recovery can
reuse that target; water at norm .8 takes 102 recorded iterations and is slower
than the failed baseline attempt. No general speed or accuracy claim follows.

`results/response_regression.json` replays all six previously independently
validated analytic derivative cases without any new ground-state SCF. Every
gradient is unchanged; the largest complete-Hessian difference is
2.379599641999022e-13 Ha/bohr^2. Wrong saddle index, inconsistent restart target,
history persistence, and zero-iteration checkpoint replay are also checked.
Production derivatives do not use finite differences.

The functional accuracy campaign is separate and remains a historically exposed
pilot, not blind validation. `protocol.json` freezes geometries, assignments,
functionals and scoring rules. It imports no semiempirical lambda and fits no
offset. `vertical.py` reads target amplitudes/orbitals but no reference excitation
energies. Failed assignment qualification must not be scored as a valid target.

A deterministic recovery extension preserves alternative stationary roots and
tries closed-orbital preparation, then orbitals from another already qualified
functional in WB97X/CAM-B3LYP/PBE0 order. Only orbitals transfer: the final energy,
gradient and response remain those of the requested functional. This extension
was introduced after observing a diffuse PBE0 ethylene root (particle overlap
.455, <r^2>=43.5 bohr^2 versus seed 4.04); no target energy selected the retry.
Intermediate attempts and partial pilot predictions are retained separately.
A partial functional comparison is not evidence for a universally improved ROKS
functional and does not change the semiempirical accuracy ledger.

AO reference contract: some archived carbonyl references use Cartesian Gaussian
AOs. Their checkpoint molecule/overlap and geometry are now checked explicitly;
occupied orbitals and frozen target vectors are projected into the common
spherical aug-cc-pVDZ calculation basis before SCF. Previously completed ground
references in that basis are reused. Acetaldehyde/WB97X qualifies after this
projection (4.180430613 eV, orbital residual 6.98e-12).

The `vertical/` archive is a timestamped partial campaign snapshot. Only a
`result.json` denotes a stationary, state-qualified prediction; alternative and
unfinished accepted checkpoints are retained for provenance, never scored.
`functional_comparison.json` leaves aggregate MAEs and the globally selected
functional unset until all required pilot cases are present. Failed early driver
attempts concern the AO/projection contract and are preserved in remote logs.
