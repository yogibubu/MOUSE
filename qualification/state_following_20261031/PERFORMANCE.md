# Orbital response performance repair, 31 October 2026

The nitropyridine preparation took 1273.55 seconds and its unsuccessful B004
continuation took 691.42 seconds on Oracle with two threads. These costs do not
meet the intended semiempirical use case. The B004 continuation reached fraction
0.7890625 and then its fixed DFT-transition fidelity reached the 0.90 boundary.
That is an identity/branch issue, not a justification for looser stationarity or
identity thresholds. The original campaign stopped at this case; its ledger
remains six matched states of fourteen and must not be described as advancing.

An isolated performance checkout on verified `oracle.sns.it` is stored at
`/home/enzo/MOUSE-orbital-performance-20261031`. It preserves the frozen campaign
sources, inputs, results and accepted checkpoints. No further DFT preparation
was launched during this performance diagnosis.

The old analytic orbital Jacobian evaluated charge and spin response separately
for every orbital perturbation and both spin channels. For 44 basis functions
and 510 rotations this meant 1020 charge and 1020 spin calls. The new code forms
both density perturbations once, evaluates the common charge response once, and
applies the full AO-pair operators to batches with matrix products. Native scalar
response implementations retain a scalar fallback. No model parameter, energy
functional, spin treatment, tolerance or state-identity gate was changed.

At the identical retained nitropyridine checkpoint, the mixed-state analytic
Jacobian took 4.13837 seconds before and 0.241612 seconds after: 17.13-fold kernel
speedup. The maximum elementwise difference of the 510 by 510 Jacobians is
8.88e-16. This is a single kernel comparison with profiling enabled, not an
end-to-end benchmark or a comparison with TDA. Results are in
`remote/performance_kernel/comparison.json`; remote raw profiles and matrices
are in each checkout's `performance_kernel` folder.

Remote regression: 23 tests passed in 9.60 seconds, covering `test_roks.py`,
`test_nonlocal_response.py`, `test_relative_pair_roks.py` and
`test_roks_trial_guard.py`. The additional batch test compares charge/spin Fock
response and the orbital Jacobian against scalar application of the same
operators. Existing independent orbital-rotation and relaxed-Hessian tests also
pass. Heavy calculations and tests were performed only on Oracle.

Remaining work: measure full-solver cost with this implementation; eliminate or
amortize expensive DFT-response calibration; resolve the B004 branch identity
without relabelling it or weakening acceptance criteria; complete the retained
14-state scope. The separate B029 bounded profile exposed failure of the initial
fixed identity gate before iteration, so that case also needs an assigned seed
repair. Neither profiling run constitutes a new matched excited-state prediction.

## Assigned endpoint repair, 31 October

The isolated assigned-state campaign on Oracle is at
`/home/enzo/MOUSE-orbital-performance-20261031`. Nitropyridine B004 is stationary
and assigned at 4.2634703236 eV (residual 5.22e-15, index 6); B029 is stationary
and assigned at 4.1531721268 eV (residual 2.56e-13, index 8). B004 parents took
1.01/4.56 seconds and the coupled state plus three ground certificates 8.10 s;
B029 parents took 1.28/7.90 seconds followed by 14.04 s. Reference preparation
is cached. The set now contains eight assigned states of fourteen. No new TBE
labels have been opened. Only one DFT root per target symmetry is available
for nitropyridine: these audits do not prove global uniqueness.

B004 was repaired under the unchanged 0.90 fidelity gate by independent NTO
initialization. Q has index four, whereas the old interpolation followed a
mixing index-three branch. CT requires an explicit assignment amendment. The
final policy uses f = |<X,u v^T>| / sigma_1(X), with X the unit-normalized fixed
projected transition. Require f squared > 1/2, occupied/virtual norms >=0.7,
local CSF overlap >=0.90 and raw projected-overlap margin >=0.05 against every
available converged same-symmetry root. Absolute active-pair anchor overlap is
a diagnostic when a fixed reference transition is supplied. This ratio is not
a many-electron state probability. A raw-overlap-majority prototype was rejected
before the remaining six predictions because it also penalized compact-space
single-pair representability. Prior policies and outcomes remain archived.
The energy functional, 50/50 weights, lambda, zero offset and stationarity
thresholds are unchanged. No improved accuracy is inferred from assignment.

The package exports `assigned_transition_orbitals`; it preserves the closed-shell
density and is invariant to ground occupied/virtual chart rotations. The revised
Oracle suite passes 26 tests in 13.07 s. Derivatives for the revised molecular
extension remain to be qualified separately from the previous five-minimum
campaign. Preparation now calibrates the reference without running the old
endpoint index cascades. An implementation repair restored that calibration-only
call after removing the old solver call; its previous sources and failed attempt
are preserved. Completed ground, response and stationary checkpoints are reused.

The subsequent v3 pipeline has also completed both adenine targets B007/B014, bringing the matched count to ten of fourteen before unsealing accuracy labels. The exact frozen v3 protocol and four newly completed result files are copied in `remote/assigned_revision_snapshot`.
