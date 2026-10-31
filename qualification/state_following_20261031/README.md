> Current revision: the assigned-endpoint solver supersedes the original stalled continuation. Ten of fourteen states are stationary and assigned. See [PERFORMANCE.md](PERFORMANCE.md) for the acceleration, explicit identity-policy amendment and remaining accuracy/derivative qualification. The current Oracle root is `/home/enzo/MOUSE-orbital-performance-20261031`; older reports below are retained generation records.

# State-following reliability experiments, 31 October 2026

All numerical work runs on verified `oracle.sns.it`, in `/home/enzo/MOUSE-state-following-20261031`, with two threads. The original publication-validation baseline, frozen state manifest, scoring labels and model are retained unchanged. These recovery experiments are exploratory numerical-method development, reported separately from the original blind baseline.

## Implemented

- Optional `trial_guard` callbacks in generalized mode following and orbital Newton reject state-changing orbital steps before evaluating them. The guards add no energy penalty and do not modify the gradient/Hessian or the stationarity/index certificates. Defaults are unchanged.
- A common-orbital singlet-overlap calculation includes the coherent spin-exchanged determinant terms and is verified against APOC in a nonorthogonal AO metric.
- A guard combines local full-CSF continuity, fixed global active-pair identity, occupied/virtual sector norms and reference-transition character. Open-orbital exchange and phases are invariant.
- An explicit `rank_one` transition metric accounts for reference configuration mixing: for a unit-Frobenius reference amplitude X, the greatest overlap with one normalized orbital pair is its largest singular value s1. Candidate fidelity is `abs(u.T @ X @ v) / s1`. Raw overlap, s1 and fidelity are all recorded. Eligibility still requires a projected norm >= 0.7, s1 >= 0.7 and relative separation `1-s2/s1 >= 0.05`. Fidelity >= 0.9, local CSF >= 0.9 and active/sector overlaps >= 0.7 are required. This is an explicitly revised assignment diagnostic, not a claim that the original raw-overlap criterion passed.
- Adaptive continuation changes only the one-body operator along F to Q. Charge/spin response is identical in both endpoints. Certified orbital Newton follows the electronic branch and reports the actual full index, which may change. The endpoint is exactly the original Q model. The original relative-pair and direct restricted-ground solves then provide the final 50/50 excitation energy; lambda remains 0.9 and no offset is introduced.

## Completed validation

Remote regression suite: **31 passed in 36.21 s**. Two additional molecular API tests passed: interruption after accepted fractions 0.2 and 0.4, disk restart at 0.6 without recomputing accepted stages, exact endpoint fingerprint and energy, and rejection of a checkpoint from a different model. The four local engine files match the tested remote source byte for byte.

- Cyclopentadiene B016: recovered. Endpoint transition fidelity 0.9459386; raw overlap 0.6977087; best attainable single-pair overlap 0.7375836. Local CSF overlaps along accepted homotopy steps exceed 0.997. Joint residual 2.18e-11; minimum absolute orbital curvature 0.00488973 Hartree; excitation 8.19727789 eV. Endpoint energy check exactly zero within recorded precision. The original method's failure remains archived.
- Benzoquinone B039: recovered. Fidelity 0.9924783; raw overlap 0.9333570; local continuity exceeds 0.998. Parent index changes 4 to 5; final joint index 6. Joint residual 4.71e-12; minimum absolute curvature 0.0114093 Hartree; excitation 7.68122729 eV. Endpoint energy check exactly zero within recorded precision.

- Aza-naphthalene B011: recovered. Fidelity 0.9965863; raw overlap 0.9512314; joint residual 7.10e-13; final joint index 8; excitation 5.27855433 eV. Parent index changes 3 to 4 to 5. Runtime 462.55 s on Oracle.
- Positive controls: diazirine B001 and furan B022 reproduce the previous excitation energies with maximum change 9.69e-14 eV.

The assignment audit passed against all available converged same-symmetry competing roots. Minimum projected-overlap margins exceed 0.05 for cyclopentadiene, benzoquinone, diazirine and furan. Aza-naphthalene has no competing root in its targeted reference subset, so exclusivity is **not tested** there. Reference coverage is incomplete in every case; these checks do not establish global uniqueness.

These are solver/identity results, not demonstrated accuracy against TBEs. Scoring labels remain unopened. The three recoveries are exploratory after observing baseline failures; the original blind results are preserved. The explicit public continuation API is available. A separate automatic v2 driver now integrates it into the 44-state campaign; validation and execution status are recorded below. Global production defaults remain unchanged.

## Basis audit

The separate audit projects each assigned DFT transition into both (i) the entire compact native Hilbert space and (ii) the native ground occupied/virtual sectors. This distinguishes missing basis directions from a poor ground-sector split and from a poor canonical seed.

Seven rejected cases remain below the original 0.7 norm gate even when the entire compact space is used: butadiene Rydberg (0.321), carbonic-acid pi-pi (0.685), diazirine Rydberg (0.486), furan Rydberg (0.283), isobutene Rydberg (0.433), triazapentalene pi-pi (0.670) and triazapentalene Rydberg (0.316). These cannot be rescued just by reassigning native orbitals. No threshold was lowered to hide these failures. A basis/model extension would be separate methodological work.

## Files

- `continuation_recovery.py`: initial raw-overlap continuation; it stops near t=0.958 for cyclopentadiene as the raw overlap crosses 0.7 despite smooth local continuity.
- `continuation_rank_one.py`: revised fidelity-based continuation, with checkpointed accepted stages.
- `audit_representation.py`: full-space versus occupation-sector audit.
- `audit_recovered_assignment.py`: completed audit against available competing reference roots; original native S0 orbitals define the fixed occupied/virtual spaces.
- Source changes: `matrix_gfn2/orbital_saddle.py`, `roks_orbital_response.py`, `roks_tracking.py`, `roks_continuation.py` (public adaptive API exported from `matrix_gfn2`).
- Tests: `qualification/test_roks_trial_guard.py`.

- `REPORT.json`: checked summary, source hashes, checkpoint tests and limitations.
- `state_following_recovery.pdf`, `.svg`, `.png`: recorded branch continuity and certified saddle indices.
- `validated_results.tar.gz`: compact archive of completed remote evidence and tested engine sources.

## Automatic v2 integration

`automatic_pipeline.py`, `automatic_worker.py` and `automatic_audit.py` implement a separate checkpointed prediction ledger at `/home/enzo/MOUSE-state-following-20261031/automatic_v2`. The protocol records source hashes and the unchanged manifest before predictions. It retains the original baseline ledger and adopts the five completed exploratory cases without rerunning them. A cached preparation check for furan passed remotely before launch.

The pipeline consumes existing references first, then resumes missing targeted references using saved ground checkpoints. Native/full-response preparation remains unchanged. Qualified spectral calibration produces the same Q operator; the public continuation API replaces the fixed-index Q search. Accepted continuation checkpoints and coupled relative-pair checkpoints are persisted. Each final joint component must pass the fixed transition guard, followed by the available-root assignment audit. Every completed result, representation failure, reference failure, timeout and ambiguous assignment is retained. No sealed numerical labels are read and no accuracy score is computed.

Launch on verified `oracle.sns.it`: PID 515911, two threads and one electronic worker, with the shared reference compute lock held by the orchestrator. Remote `automatic_pipeline.log`, `automatic_v2/progress.json` and individual worker logs record actual progress. Launch is not completion: the final 44-state ledger and accuracy validation remain pending.

First live checkpoint after launch: 17 states recorded from existing evidence (five adopted matched states, seven representation failures and five unassigned states). The pipeline advanced to targeted reference preparation for Bimane_Syn. These counts are operational coverage, not 17 new predictions.

## Reduced campaign, 31 October 2026

The user requested fewer molecules and a focus on resolving the method. The expanded 44-state campaign is stopped. On inspection its orchestrator had already exited at Bimane Syn's canonical-ordering precheck; no orchestration or preparation process remained active. The stale `running` progress entry was corrected. Native preparation, the DFT reference and the qualified full-response B015 parent are retained.

`reduced_scope.json` defines nine primary molecules with 13 valence/CT states, plus the already completed benzoquinone stress case: ten molecules total. This is a documented reduction after observing the original baseline, not a new blind preregistration. All original results, including out-of-scope outcomes, remain archived. Work is currently focused on Bimane before starting any additional molecule.

`probe_spectral_ordering.py` evaluates the exact original spectral operator in a separate diagnostic directory. Its original canonical occupied/virtual ordering check gives a gap of -0.01043974712944426 Hartree (about -0.2841 eV). The probe records that inversion and allows only the diagnostic export to continue. This does not change any operator coefficient, exchange scale, response kernel or offset, and does not relax production criteria. `probe_ordering_worker.py` then tests continuation, fixed state identity, the coupled stationary point and certified restricted ground minima. Removing a preliminary canonical-order check is acceptable only if these stronger physical endpoint checks succeed. Results are exploratory and no excitation reference labels are used.

### Bimane model diagnosis and revised one-body candidate

The original Q diagnostic reaches a smooth assigned branch through t=0.8875 (transition fidelity 0.9109), then approaches the fixed native occupied-sector floor. Its independently optimized Q ground state is a certified minimum: residual 1.22e-15, smallest curvature 0.13346 Hartree, index zero. Nevertheless the projected target DFT transition norm in that relaxed ground chart falls from 0.92447 in the native chart to 0.31970. The diagnostic therefore does **not** justify bypassing the canonical-order or identity gates in production. The accepted excited-state checkpoint and the certified Q ground checkpoint are retained.

A separate exploratory candidate, `ritz_projection_probe`, replaces the partially mapped spectral levels by the full DFT ground-state Fock projected through the existing isometric embedding: `F_small = L.T @ S_DFT @ C_DFT @ diag(e_DFT) @ C_DFT.T @ S_DFT @ L`. The added one-body operator is `F_small - F_native`. Full AO-pair charge/spin response, lambda=0.9, 50/50 weights and zero scalar offset are unchanged. This is an explicit model revision, not recovery of the original Q endpoint. No target excitation energies enter its construction. The builder records input/source hashes and verifies exact reconstruction of the projected Fock before prediction.

`build_ritz_reference.py`, `ritz_projection_worker.py`, and `ritz_projection_audit.py` implement that candidate. The worker retains continuation and joint checkpoints and saves each certified restricted ground state separately. Its Bimane test and two molecular controls have completed successfully on Oracle. Energy accuracy and nuclear derivatives of this revised profile remain unvalidated.

### Completed repair evidence

The complete projected-Fock candidate passes the unchanged numerical, identity and restricted-ground minimum checks for Bimane B015, diazirine B001 and furan B022. Their new-profile energies are 3.91145521, 4.84050618 and 8.30266621 eV, respectively. These changed energies are not evidence of increased accuracy; labels remain unopened. Bimane has no competing root in its targeted reference subset, so global uniqueness is not tested there.

For Bimane, the joint residual is 5.11e-15, minimum absolute coupled curvature 1.9120e-4 Hartree, and both component gaps are positive (3.99300844 and 3.82990198 eV). All three ground references have index zero, positive smallest curvature and residual below 1.5e-11. The revised relaxed-Q chart retains a DFT transition projection norm of 0.92529, compared with 0.31970 for the old partial-level model.

`select_consistent_reference.py` records a conservative molecular rule: retain the original qualified spectral operator when its existing canonical gap check is positive; otherwise use the complete projected ground Fock. Selection is made once for the molecular anchor, without excitation labels or error-based choice. The response kernels are checked byte-identical. The rule selects the original model for the five previously completed molecular cases and the revised model for Bimane; all six selected results are numerically qualified and existing work is reused. This selection is verified as a candidate workflow; it is not installed as a global production default or evidence of universal energy accuracy.

`REFERENCE_REPORT.json` collects the six selected results, all three full-projection controls and explicit limitations. `consistent_reference_repair.pdf`, `.svg` and `.png` visualize the recorded sector continuity and ground-chart consistency. `candidate_evidence.tar.gz` archives the completed remote JSON evidence. The enlarged 44-state pipeline remains stopped; no further molecular jobs were started after these two controls.

## Continuation on the remaining reduced cases

The reduced 14-state/10-molecule campaign is launched with source hashes frozen. See `reduced_validation/README.md`. A separately confirmed correction distinguishes true TDA A from the static symmetric-density response for nonlocal exchange. PySCF operator tests pass for alpha=1 and 0.225. Six completed static controls are archived and reused, not recalculated from scratch; OO results are unchanged. The old full-response TDA comparisons in the manuscript require regeneration before use. No numerical excitation reference labels have been opened.

The reduced campaign now uses a validated single-vector, Ritz-checkpointed PySCF reference eigensolver. See the reference amendment in `reduced_validation/README.md`; completed SCF and excited predictions are retained, numerical response tolerance and Hamiltonian are unchanged.
