# Response and REKS solution audit — 30 October 2026

## Decision

Keep the requested fixed 50/50 complete-response/spectral-response energy mean as an experimental baseline. No fitted energy offset, orbital-gap rescaling or molecule-dependent weight is introduced. None of the new alternatives establishes a universal n→pi* correction. No default is changed.

## Prospective test D

The frozen test qualifies all three molecules. MAE: native 0.619169 eV; complete 0.318641; spectral 0.127708; fixed mean 0.170636. The original gate fails because the mean does not beat the spectral component. The optimized-weight diagnostics were not applied.

## New physical hypotheses tested

1. DFT levels and XC response from the same stationary DFT ground density. Independent PySCF response audits pass (largest matrix error below 8e-15). On six exposed states the fixed-mean MAE changes from 0.128922 to 0.193593 eV. Reject this as a general accuracy correction.

2. REKS(2,2) fixed-orbital ground correlation, with the published fixed interpolation delta=0.4 and six spin-polarized microstates. Across the tested pairs the maximum ground lowering is 0.00012061 eV. This fixed-orbital screen does not support a ground-correlation shift of several tenths of an eV. It is not a full REKS prediction.

3. Variational SA-REKS(2,2), sharing orbitals between PPS and OSS and optimizing the active occupations. Analytic electronic response includes the occupation Schur complement. Scalar energy checks verify its first and second orbital derivatives. Higher electronic saddle indices are explored when the minimum loses the tracked state. All failed or changed-state roots are retained.

| Exposed subset | SA-REKS spectral MAE | Original spectral MAE | Original fixed-mean MAE |
|---|---:|---:|---:|
| carbonyl_B | 0.079764 | 0.262595 | 0.053547 |
| C_exposed | 0.164352 | 0.168143 | 0.246428 |
| D_exposed | 0.211913 | 0.127708 | 0.170636 |

All values are eV. These SA-REKS results are exposed-data diagnostics, not a second blind D test.

The carbonyl improvement fails to transfer. Acrolein has a qualified index-1 SA-REKS gap of about 4.171 eV against 3.723 eV. The qualified index-2 uracil n→pi* root is about 5.527 eV against 4.935 eV. The low-energy uracil minimum and several other apparent roots lose CSF identity and cannot count as predictions. SA-REKS is therefore not promoted.

## Consequences for development

The evidence does not support solving the residual error by changing the mean, a universal scalar lambda, stationary-density kernel replacement, or the tested two-orbital ensemble alone. A further candidate must change the density/orbital-dependent functional consistently and pass a new frozen independent test; the present data cannot establish which new functional will succeed. Keep ground and excited energies from the same functional and retain explicit spin adaptation and state tracking.

Existing native and frozen shell/local-tensor ROKS analytic nuclear Hessians remain separate from these nonlocal research profiles. The complete response, spectral response, stationary-density and SA-REKS prototypes are vertical-only. The analytic electronic Hessian in the SA-REKS solver is not an analytic nuclear Hessian. No new production nuclear derivative implementation is delivered by this screen.

Sources: [REKS analytic response](https://doi.org/10.1063/1.4994542), [DFTB REKS implementation](https://doi.org/10.1021/acs.jctc.9b00132), [official DFTB+ REKS description](https://dftbplus-recipes.readthedocs.io/en/latest/reks/introduction.html).

All electronic work ran on oracle.sns.it. Existing references and completed checkpoints were reused. See the JSON summaries and solution_provenance.json for exact records and artifact hashes.
