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
