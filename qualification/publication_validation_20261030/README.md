# Publication-validation campaign, 30 October 2026

Remote root: `/home/enzo/MOUSE-publication-validation-20261030` on verified `oracle.sns.it`.
Heavy work is remote only. Two threads; one serial electronic worker. Numerical reference labels are not copied back or opened for scoring.

Completed:
- Published dedicated xtb4stda + std2 comparison on identical archived geometries: all 19 transitions matched by cross-AO transition identity. MAE 0.454009 eV versus MOUSE 0.182527 eV; previous prospective four-state MAEs 0.277975 versus 0.303628 eV.
- TD-DFTB2/mio-1-1 numerical spectra for 18 molecules; DeltaDFTB2 numerical completion for 16. Assignment remains mandatory before accuracy reporting. Two DeltaDFTB SCC failures are retained after a documented DIIS recovery.
- Three-molecule CCSDR(3) internal-coordinate comparison, vertical-anchor-only algebraic weight sensitivity, provisional frequency-order HNO harmonic comparison.
- Exposure-audited holdout manifest: 44 states / 30 molecules. 12 n-pi, 12 pi-pi, 8 CT, 12 Rydberg. CT quota shortfall retained. Manifest SHA-256 `db5edebcde1256888c6d6f73e7838f0c67e8c3e41cc47d54f088d08edc00ee93`.

Running/queued:
- `blind_reference.py` prepares label-free PBE0/aug-cc-pVDZ ground references and up to 40 TDA roots, recording individual root convergence and incomplete 20 eV coverage.
- `blind_mouse.py` waits for the reference batch's `blind/references/compute.lock`, then runs native seed qualification, unchanged projected full/spectral response, stationary ROKS parents, relative-active-pair solution and direct restricted ground references. It reports missing symmetry assignments, compact-basis failures, nonconvergence and timeouts.
- `preflight_blind_mouse.py` passed remote imports, public assignment, transition normalization and API extraction; it stops before electronic evaluation and does not establish numerical solver success.
- All workflows preserve checkpoints. No blind score may be reported before the all-state prediction ledger and coverage are audited.

Public assignments use state symmetries and their ordinal in the original public metadata; numerical TBE values are not exported or read by the predictor. Degenerate groups with unresolved correlations remain unassigned. PySCF X vectors are exported with unit normalization for consistent projection gates; original reference spectra remain immutable.

Inspect remotely:

```sh
ssh oracle '/home/enzo/nano-MATRIX-current/.venv/bin/python /home/enzo/MOUSE-publication-validation-20261030/collect_status.py'
```

`STATUS.json` and the fetched `remote_prechecks/` files are snapshots, not live progress.

Outstanding: qualified TD-DFTB/DeltaDFTB assignments, blind predictions/coverage audit, training-calibrated TDA-GFN2, full lambda/anchor/weight response tests, additional high-level frequency references and mode confirmation, and complete cold/warm derivative/memory cost scaling. The manuscript includes only completed comparisons and labels these limitations explicitly.

## Scheduling repair after the first reference batch

The initial batch-wide lock delayed all OO predictions until every DFT spectrum finished. `serial_blind_pipeline.py` replaces only orchestration: it adopts the current reference worker until the original one-hour limit, preserves all ground/spectral checkpoints, consumes existing complete/partial references first, then runs each new reference immediately followed by its MOUSE prediction. All work remains serial on Oracle with two threads. Frozen reference and prediction worker sources, geometry/state manifests, energy model, assignment gates and scoring labels are unchanged. The scheduler's own protocol and transition record are in `blind/pipeline_protocol.json` and `blind/pipeline_progress.json` remotely.

## Recovery on 31 October 2026

The first MOUSE workers stopped before native calculations because an extracted legacy preparation helper expected `edft_holdout_F_protocol.json` in the new campaign root. That filename now links to the actual frozen blind MOUSE protocol; old fitting labels or old cases are not imported. The MOUSE source, model, weights and numerical gates are unchanged. An end-to-end Diazirine B001 prediction now qualifies at 4.241949435486163 eV (joint residual 2.07e-15); its Rydberg target B044 fails compact representation and remains in the denominator. No TBE scoring has been performed.

`recover_blind_pipeline.py` first consumes already available references and stops on unexpected MOUSE execution errors. `targeted_reference.py` prepares remaining DFT reference responses only in the fixed public target symmetries, through the required ordinal root. It reuses every converged PBE0 ground checkpoint and preserves previous complete/partial spectra. This change to reference scheduling is recorded in `blind/targeted_protocol.json` before calculation. Same reference functional/basis, convergence thresholds and two-thread serial policy; a targeted spectrum is not a complete 20 eV spectrum. The twelve original full-spectrum timeouts are retained in the audit.
