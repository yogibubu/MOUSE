# Final frozen-model pi→pi* tests

Registered in a474e68 before predictions; candidate freeze 467a9e5 and public
implementation ecb6644 remain unchanged. Furan (^1A1, historical B022) and
uracil (historical pi_pi) add two distinct pi→pi* tests. Native ground and
PBE0 reference inputs are reused. Uracil's missing bare Coulomb tensor was
reconstructed once on verified Oracle in its original PBE0 embedding; the
RSH integral cache uses a different embedding and was not mixed with it.
All subsequent electronic preparation, optimization and validation ran on
verified MAC-STUDIO. Twenty-one transferred input hashes pass verification.

| Target | Reference / eV | OO-CID / eV | Signed error / eV |
|---|---:|---:|---:|
| Furan pi→pi* | 6.530 | 5.804329 | -0.725671 |
| Uracil pi→pi* | 5.381 | 6.080116 | +0.699116 |

Both converge and pass fixed state-assignment gates, but both fail the
unchanged 0.5 eV accuracy gate. Their MAE is 0.712393 eV, against 1.313121
eV for matched native-orbital TDA. No fitted offset, weight or state-specific
selection change was made. Opposite signed errors separated by 1.424787 eV
cannot both fit inside +/-0.5 eV under any common additive offset.

The full dataset now contains 20 states in 18 molecules: MAE 0.293336 eV,
maximum error 0.725671 eV. The seven predictions after candidate freeze give
0.424590 eV MAE. The pooled maximum and post-freeze mean fail the fixed gates;
all twenty states remain scored. These historically exposed molecules are
not a blind benchmark. The 13 n→pi* states give 0.268535 eV MAE and the seven
pi→pi* states 0.339395 eV MAE. The prior 15- and 18-state archives are retained
unchanged; the larger sample must not be represented by the earlier passing
maximum alone.

Two complete checksummed portable models are included. Fresh-process energy
and restart checks block PySCF imports, new SCC calls and TDA input reads.
Exact coherent determinant-sum checks validate S²=0. The mathematical model
and 26 audited public source files are unchanged. These are energy-only tests;
no new analytic gradient/Hessian qualification is claimed. The original five
states in three molecules remain the derivative qualification set. Both
geometries match existing archived Jmol structures within 1e-8 Angstrom.

score_last_two.py reproduces raw and combined CSV/JSON tables without fitting.
TRANSFER_AUDIT.json hashes every bundled artifact; the remote inventory records
preparation sources, retained checkpoints and 21 input-file checks. All raw
controls are retained; the better full-SI control for furan is not substituted
for the frozen projected primary prediction after seeing its reference error.
