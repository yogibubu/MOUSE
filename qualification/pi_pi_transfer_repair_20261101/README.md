# Screened symmetry-augmented MOUSE OO-CID

All electronic calculations and numerical validation in this directory run only on verified Oracle or MAC-STUDIO, never on the laptop. The laptop edits, inspects and transfers artifacts.

The selected candidate resolves all seven exposed pi-to-pi* targets while keeping thirteen n-to-pi* controls within 0.5 eV. Its fixed settings are F/Q weights 50/50, exchange-response lambda 0.9, and zero excitation-energy offset. Reference orbital gaps are not scaled by lambda.

| Set | States | MAE / eV | Maximum absolute error / eV |
|---|---:|---:|---:|
| pi-to-pi* | 7 | 0.179792 | 0.349409 |
| n-to-pi* | 13 | 0.249674 | 0.497948 |
| All | 20 | 0.225215 | 0.497948 |

All twenty labels were exposed during development. This is a candidate tested on this finite planar valence set, not a blind or universal-accuracy result.

## One common construction

Retain the optimized singlet ROKS targets, their paired doubles, and the original two-occupied/two-virtual frontier doubles in both reflection parities. Add reflection-even singlet doubles from the five-occupied/five-virtual frontier, with both exact singlet recouplings of every four-open-shell occupation. Reflection and MO parities are certified with nuclear geometry and exact Gaussian AO overlaps, never with state-family or excitation-energy labels. Paired-double duplicate removal retains the original overlap threshold. The list is frozen at the anchor, including during out-of-plane displacement.

Every state receives the same statically screened two-body interaction and consistently normal-ordered one-body operator. The native spin-polarized diagonal energy and exact four-spin blocks are retained. A moving stationary-ground projector precedes the CSF generalized eigenproblem. No per-molecule parameter or energy offset is used. `protocol.json` documents the development sequence and scope.

The unconstrained five-pair diagnostic, earlier kernel variants and all completed results remain in the archive; none is silently substituted for the selected candidate.

## Public implementation

`PreparedNOSIModel` recognizes `mouse.nosi.screened_even_cid.v1` automatically. The existing `nosi_50_50` LINK selector accepts the new bundle; legacy selected-CID bundles retain their original behavior. Numerical bundles use relative filenames and SHA-256 verification. Runtime imports no DFT backend, reads no backend TDA input and invokes no new ground SCC driver. The native stationary ground and ROKS orbitals still respond to nuclear changes.

Analytic gradients and Hessians contain first and second implicit orbital/KKT response, the moving ground projector, all diagonal and spin terms, the metric, and generalized eigenvector response. Frozen screening and auxiliary orbitals are transported in the anchor's co-rotating orthonormal AO frame. Finite differences are validation only.

The complete 15x15, 27x27 and 36x36 molecular Hessians cover the two Criegee states, furan and uracil. Two independently reconverged energy directions, including an out-of-plane displacement, at two step sizes give maximum Richardson gradient and curvature errors of 1.42e-10 Ha/bohr and 3.72e-7 Ha/bohr². Raw Hessian asymmetry is below 4.65e-14 Ha/bohr². A cold direct public-API Hessian agrees with assembled matrices within 2.16e-9 Ha/bohr². All twenty fresh-process energy/spin/restart checks pass. The LINK pi state also passes accepted-only restart, coherent following, and displaced-trial rollback checks.

Exact acceleration uses canonical occupation bitstrings for the auxiliary block, cached NumPy greedy contraction programs, omission of exactly zero jet terms, and an invariant auxiliary Gram block. The molecular mixed-response column changes by at most 4.11e-15 Ha/bohr². Completed pre-acceleration analytic columns are preserved and reused.

Prepared energy calls take about 1.7--9.0 s with two BLAS threads, excluding loading/preparation. The dense direct two-state Criegee Hessian takes 418 s and about 1 GiB. These timings do not establish a vertical speed advantage over matched TDA. Parallel columns qualify the full matrices separately; they are not substituted for a serial API timing.

## Portable example (remote hosts only)

With MOUSE and its native extension installed:

```python
from pathlib import Path
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.nosi_surface import PreparedNOSIModel

root = Path('qualification/pi_pi_transfer_repair_20261101/portable_examples')
model = PreparedNOSIModel(root / 'independent/Criegee_Intermediate/model.json')
symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
calculator = GFN2Calculator([symbols[int(z)] for z in model.anchor.atomic_numbers],
                           temperature_kelvin=0.)
result = model.evaluate(calculator,
                        model.anchor.coordinates_bohr / ANGSTROM_TO_BOHR,
                        properties=('energy', 'gradient', 'hessian'))
```

LINK uses `native_excited_method='nosi_50_50'` and the checksummed `native_nosi_model_path`. The second Criegee model state is its pi-to-pi* target. Only accepting a result advances the restart generation.

## Provenance

- `2d59ab8`: selected exposed-data candidate freeze.
- `81897cf`: public extended configurations and complete analytic response.
- `e572d50`: exact contraction and metric acceleration.
- `selected_predictions.json` / `selected_summary.json`: every final value and gate.
- `results/`: raw completed predictions, runtime/spin checks, analytic columns, finite-energy checks and executed-source evidence.
- `portable_examples/`: representative self-contained numerical bundles.
- Large operator checkpoints and the remaining bundles stay on the verified remote host, with SHA-256 manifests.

Earlier phthalimide minima/Hessian and published-method comparisons belong to separately identified earlier model generations. They are not claimed as validation of this final energy. Broader blind transferability, final-model equilibrium structures/frequencies and end-to-end size scaling remain outside the present finite test.
