# Further reduction of the n-pi and pi-pi errors

**Subsequent validation:** the frozen profile fails on five additional molecules.
See the [generalization report](generalization/REPORT.md): paired-cohort MAE
1.354 eV with the correction, versus 0.446 eV for original SE ROKS.
The favorable uracil/thymine results below do not establish a general model.

29 October 2026. This extends the initial, unfitted radius-ratio study in
EXTENT_REPORT.md. All numerical work ran on verified `oracle.sns.it`, using
the saved molecular states and DFT references. No new electronic states were
needed for the calibration and its molecular API verification.

## Selected experimental PBE0 profile

The occupied-reference transformation now accepts an explicit positive power:

```
w_i = (sigma_SE[i] / sigma_DFT[matched_i])**p
epsilon_effective[i] = epsilon_anchor + w_i*(epsilon_DFT[i]-epsilon_anchor)
delta_gap(i,a) = lambda0*(epsilon_effective[a]-epsilon_effective[i]) - gap_SE(i,a)
```

The primary candidate fixes **p = 1.5** and fits **one** common spectral factor
to the two uracil ROKS excitation energies, giving
**lambda0 = 0.8706205766726199** for PBE0/aug-cc-pVDZ. The power is a simple
empirical candidate, not a first-principles derivation. The one fitted scalar
replaces the previous arbitrarily fixed 0.9 for this experimental profile.
It is shared by n-pi and pi-pi and applied identically in ROKS and TDA.
The reference spectrum is transformed before TDA diagonalization; SE orbitals,
spin polarization, relaxation and response kernels are retained.

The same parameters are reused for PBE0/aug-cc-pVTZ without a basis refit.
They are not universal parameters for other DFT functionals. B3LYP requires
its own calibration; outputs obtained by transferring the PBE0 factor to
B3LYP are diagnostics, not the selected B3LYP model.

## Energies and held-out errors

With PBE0/aug-cc-pVDZ reference energies, all values in eV:

| Molecule and role | State | Previous ratio model, p=1 and lambda0=0.9 | New model | QUEST | New error |
|---|---|---:|---:|---:|---:|
| Uracil, calibration | n-pi | 5.505476 | 4.957719 | 4.935 | +0.022719 |
| Uracil, calibration | pi-pi | 5.633638 | 5.360979 | 5.381 | -0.020021 |
| Thymine, excluded from calibration | n-pi | 5.447407 | 4.873312 | 4.980 | -0.106688 |
| Thymine, excluded from calibration | pi-pi | 5.560449 | 5.354862 | 5.279 | +0.075862 |

The thymine MAE falls from **0.374428 to 0.091275 eV**, with the correct state
ordering. The uracil training MAE is 0.021370 eV; it is explicitly a fitted
result. With the same parameters and aug-cc-pVTZ, thymine gives 4.896341 and
5.371797 eV, with errors -0.083659 and +0.092797 eV.

The calibrated ROKS parameters also enter the GFN2 functional TDA matrix.
On uracil, its trusted 42-configuration values become 4.366456 and 6.195674 eV.
This profile was calibrated to ROKS, so those values do not constitute an
algorithm-neutral accuracy benchmark. Both methods still use the same GFN2
orbitals and transformed DFT spectrum. The SE TDA kernel has distinct residual
errors and is not corrected by fitting only the ROKS spectrum.

## Cross-validation and overfitting control

Repeat the one-scalar fit in the reverse direction: calibrate on thymine only,
then predict uracil. Aggregate the four predictions with each molecule excluded
from its own fit. The exploratory leave-one-molecule-out MAE is **0.056027 eV**,
with maximum absolute error **0.106688 eV**, compared with 0.392993 and
0.570476 eV for the previous unfitted ratio model.

For comparison, allowing both power and scale to fit the two uracil states
gives p=1.571566 and lambda0=0.876675, exactly interpolating those two energies.
Its thymine MAE is worse, 0.119053 eV, and its aggregate reverse/forward
leave-one-molecule-out MAE is 0.105287 eV. Exact reproduction of the training
pair therefore does not justify selecting the two-parameter model.

The one-scalar model is retained as the simpler candidate. The candidate
comparison and cross-validation involve only two related molecules and are
exploratory. Neither the fitted uracil values nor this small model-selection
exercise establish transferable accuracy across chemical families. Further
independent molecules are needed before changing production defaults.

## Code, checks and retained results

`with_occupied_extent_scaling(..., power=1.5)` uses the same immutable spectrum
object for both ROKS and TDA. The default remains power=1 for compatibility;
the calibrated profile is explicit and recorded in
`results/extent/calibration/profile_pbe0_uracil.json`.

`calibrate_extent.py` retains raw model inputs, fit parameters, both fitting
directions and all per-state errors. `verify_calibrated.py` compares the
predicted formula to actual molecular ROKS API outputs for uracil/thymine with
both PBE0 bases. `run_tests.py` records 35 passing tests, including fractional
powers, common configuration corrections, energy-origin invariance and
derivative guards. Hashes are in provenance.json.

Reproduction in the existing Oracle validation environment:

```
python calibrate_extent.py
python compare.py --reference-energy-scale 0.8706205766726199 --extent-policy basis_compactness --extent-power 1.5
python extent_holdout.py --reference-energy-scale 0.8706205766726199 --extent-power 1.5
python verify_calibrated.py
python run_tests.py
```

Use the interpreter and thread limits documented in REPORT.md. Checkpoints
are loaded without restarting completed electronic jobs. The correction remains
an energy model: analytic nuclear response of the DFT energies and orbital
radii has not been implemented, and the API does not silently substitute the
SE Hessian for the missing hybrid Hessian.
