# Three-molecule extension of the frozen selected OO-CID model

Registered before these calculations as `f8a38e3`; parent candidate freeze
`467a9e5`, public implementation `ecb6644`. No weights, exchange scaling,
offsets, configuration-space selection or assignment thresholds were changed.
All electronic work ran on verified MAC-STUDIO; original prepared inputs were
exported from verified Oracle and their 31 hashes checked after transfer.
The laptop only orchestrated, inspected, transferred and built figures/TeX.

| Molecule | Reference / eV | OO-CID / eV | Absolute error / eV |
|---|---:|---:|---:|
| Pyrimidine | 4.451 | 4.637126 | 0.186126 |
| Cyclopentadienone | 2.951 | 3.411733 | 0.460733 |
| Pyranone | 4.016 | 4.512496 | 0.496496 |

All three are n→pi* transitions, selected for diazine substitution pattern,
conjugated cyclic ketone and oxygen-heterocycle environments. All assignments
and individual <=0.5 eV error checks pass. **The new three-state MAE is
0.381118 eV and exceeds the fixed 0.3 eV mean-error gate.** Matched native-orbital
TDA gives 0.462568 eV MAE. All 18 states together give 0.246774 eV MAE and
0.497948 eV maximum error; the five predictions after freezing the candidate
give 0.309468 eV MAE. Pooling development states must not obscure the worse
new-cohort mean. These historical molecules are not a blind validation set.

Complete portable models are bundled for all three cases. Fresh-process
energy/restart checks block DFT imports, SCC calculations and TDA inputs;
coherent determinant sums independently pass the exact S²=0 singlet check.
Only energies were newly validated here: the original five-state/three-molecule
analytic gradient/Hessian qualification is unchanged. All three coordinates
match the existing archived Jmol structures to 1e-8 Angstrom.

`score_additional_three.py` rebuilds the three-state and combined eighteen-state
tables without fitting. `selected_extension.py` has the exact AST body of the
frozen matrix extraction, with only loop-skip changed to function return.
`TRANSFER_AUDIT.json` records file hashes; `additional_three_remote_inventory.json`
records raw input, preparation-source and retained checkpoint provenance.
The frozen initial 15-state `results/` and its audit remain unchanged.
