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

The `vertical/` archive now contains the complete 18-calculation campaign. Only a
`result.json` denotes a stationary, state-qualified prediction; alternative and
unfinished accepted checkpoints are retained for provenance, never scored.
`functional_comparison.json` reports all four pilot and two confirmation
states for each functional, selecting WB97X by pilot MAE alone. Failed early driver
attempts concern the AO/projection contract and are preserved in remote logs.

Final numerical/accuracy qualification
--------------------------------------
The final linear tolerance is 0.1 times the unchanged nonlinear tolerance;
`results/final_solver_linear_tolerance/` retains the repeated 6/6 recovery checks.
`results/range_cache_validation.json` compares native/cached exact J/K, orbital
energies, residuals and Jacobian actions, including native fallback at zero cache
budget. `results/range_cache_response_regression.json` preserves the complete
CAM-B3LYP analytic Hessian within 3.1e-13 Ha/bohr^2. The independent WB97X water
checks, both before and after cache activation, are in `results/WB97X*_response/`.
Cached WB97X independent gradient/Hessian-vector errors are 1.67e-9 and 1.95e-9.

`results/range_cache_benchmark.json` uses identical qualified butadiene/WB97X
orbitals, two threads and alternating native/cached orbital actions. Median
action time decreases from 3.921 to 1.675 seconds (2.34x); setup is included
separately and 460,660,368 integral bytes are cached. This is one action benchmark
under concurrent host load, not a total-run/TDA comparison or memory scaling study.

All 18 calculations satisfy the same residual and frozen-target identity criteria.
WB97X reduces pilot MAE from PBE0's 0.715 to 0.547 eV and confirmation MAE from
0.295 to 0.109 eV. Its furan error is 1.121 eV (PBE0: 0.626 eV); improvement is
not uniform. No global pi-pi accuracy cure or blind validation is established.
The functional choice remains a candidate for independent testing; the separate
semiempirical ledger and its parameters are unchanged. The figure and standalone
scorer reproduce the complete results. Logs preserve initial driver errors and
checkpoint/continuation attempts. Production code contains no molecular branches.
