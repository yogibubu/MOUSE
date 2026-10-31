# Portable multi-CSF integration qualification, 31 October 2026

Electronic work ran on oracle.sns.it, verified before launch, with two threads
and a shared compute lock. No electronic calculation ran on the laptop.
MAC-STUDIO (Mac Studio M3 Ultra, Mac15,14) was verified as a fallback during an
Oracle interruption; only source files were staged there, with no job launched.

## Completed

- All 27 furan excited-total analytic Hessian columns agree with independently
  reoptimized finite differences; maximum error 2.2852e-8 Ha/bohr². Completed
  columns and displaced points are saved individually and resumed.
- The opt-in public `multicsf_50_50` route reproduces the research energy
  exactly, its gradient within 1.4e-17 Ha/bohr and Hessian within 1.34e-14
  Ha/bohr². Cold evaluation including the analytic Hessian took 98.99 s;
  restoring the accepted state and its STORED Hessian took 0.613 s. This is
  reuse of computed derivatives, not a fresh-Hessian speedup measurement.
- Both parent overlaps at a displaced point exceed 0.99998. Direct scan and
  geometry-optimizer dispatch, portable model relocation, immutable accepted
  history and detection of corrupt model/checkpoint inputs pass. Instrumented
  surface evaluation makes no SCC, DFT or TDA call.
- 43 core/legacy/coherent-APOC tests plus 3 separate new LINK contract tests
  pass. XML and logs identify both runs without double counting.
- Diagonal exact-exchange response lambda 0.85/0.9/0.95 is tested on the three
  rank-two pi-pi cases, with the one-body DFT anchor, semilocal response and
  bare CSF couplings held fixed. Maximum changes from 0.9 are 0.063, 0.070
  and 0.085 eV. The production value remains 0.9.
- Algebraic vertical weights 0.4/0.5/0.6 are reported, without claiming their
  off-anchor derivatives. The 50/50 candidate remains fixed.
- Scaling all subleading fixed-reference NTO singular values exposes a
  phthalimide rank boundary at scale 0.90248. Switching rank changes its
  vertical energy by 0.923 eV. Rank is frozen along each PES; these artificial
  reference perturbations are not a substitute for actual recalibrated DFT
  anchors or independent validation.

## Reproduction and archives

`export_model.py` reuses stationary parent checkpoints and matching ground
certificates. Only absent ground certificates are newly solved and retained.
`validate_link.py`, `regression.py`, `lambda_sensitivity.py` and
`test_multicsf_all_columns.py` identify the executed work. The last driver is
in the adjacent pi_pi_repair qualification folder.

`remote/multicsf_integration/TRANSFER.json` records artifact hashes, all runtime
Python hashes and the compiled native extension hash. Local/remote hashes of
all 12 modified implementation files match. A complete portable furan model
is archived. Large phthalimide tensors and duplicate model copies remain on
Oracle, with their hashes recorded. Original failed setup/export attempts are
retained as logs; the angstrom/bohr conversion now uses the package constant.

Open scientific gates are actual DFT-anchor recalibration, new independent
validation, additional molecular Hessians and physical structure/frequency and
complete cost/memory comparisons. These results do not establish a universal
accuracy advantage or an end-to-end speed advantage over TDA.

## Additional qualification

`anchor_rank.py` freezes six actual reference recalculations before inspecting
results: internal Cartesian scales 0.98 and 1.02 for phthalimide, furan and
cyclopentadiene. PBE0/aug-cc-pVDZ/grid-3 ground orbitals and two same-symmetry
TDA roots are recomputed; identity requires coherent cross-geometry transition
overlap above 0.7 and competitor margin above 0.05, consistent with the original
symmetry ordinal. A fresh direct native GFN2 S0 minimum defines the compact
projection. The original fixed compact NTO rank is compared with the recalculated
rank. This is an anchor/rank diagnostic, not a recalibration of the full F/Q
response model and not an independent excitation-energy validation.

`qualify_cpd_hessian.py` checks all 33 Cartesian columns of the cyclopentadiene
excited-total analytic Hessian through the public multi-CSF interface, using
independently relaxed plus/minus displacements and saved ground/state seeds.
The 2e-4-bohr displacement and tolerances are frozen in `protocol.json` before
calculation. The first cold Hessian cost and peak process RSS are recorded;
this is not a complete size-scaling benchmark or comparison with TDA.

Both drivers assert Oracle identity, cap numerical libraries at two threads,
share the existing compute lock and reuse finished checkpoints. No calculation
from these additional tests runs on the laptop. The cyclopentadiene check is complete: 33/33 columns, maximum error
1.2964e-7 Ha/bohr². All reported errors were independently reproduced by
lightweight reduction of the transferred checkpoints. Both stored molecular
Hessians pass acoustic and rotational-covariance checks. All six actual DFT/native projection rank checks are now complete and retain
rank two. The full F/Q anchor-energy sensitivity gate remains open.

The unseeded first DFT response was interrupted after more than nine minutes
without a completed spectrum; its converged ground checkpoint and log are
retained. `anchor_rank_seeded.py` resumes that ground and initializes the same
two-root response with the transported original transition plus conventional
guesses. It changes no Hamiltonian, root-identity rule or tolerance.
`audit_anchor_projection.py` separately reproduces the original compact
projection and distinguishes transported-anchor from relaxed-native subspaces.

## Completed actual reference/rank test

`anchor_rank_symmetry.py` is the final executed driver. Its exact irrep pair
block removes forbidden-sector null-space components from restarted Ritz
vectors. Two-vector Davidson increments, a 1e-20 linear-dependence threshold
and positive-root checkpoints preserve the original 1e-7 response-residual
criterion. A separate full-operator response action certifies both roots.
Earlier unfinished/uncertified solver attempts remain in the archive and are
not accepted predictions. Ground checkpoints are preserved and reused.

All six +/-2% Cartesian-dilation points pass identity and compact-projection
gates. Effective native pair numbers are 1.5924--1.5982 (phthalimide),
2.0071--2.0135 (furan) and 1.9850--1.9927 (cyclopentadiene); rank remains two.
The minimum transition overlap is 0.9962, margin 0.9724 and compact norm 0.8256.
The largest atom shift is 0.06883 A. Reprojection of the original three
transitions reproduces them within 5e-16; co-rotating transported original
subspaces and freshly relaxed native S0 differ by at most 1.33e-4 in effective
pair number. These tests concern one local geometry direction, not fully
recalibrated OO energy models or an independent validation set.

The final archive verifies 185 artifacts and eight executed Python sources
against their local copies. The shared native-extension hash is unchanged.
All new electronic work finished on Oracle. Python 3.11.15, NumPy 2.3.5 and
PySCF 2.14.0 identify the reference runtime. Complete SCF backend checkpoints
remain on Oracle with hashes in the transfer manifest.
