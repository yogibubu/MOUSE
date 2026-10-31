# Frozen DFT excitation plus semiempirical relaxation

**Outcome:** the no-fit hybrid improves the six-state previous generalization cohort MAE from 0.446220 to 0.211504 eV, and the full ten-state MAE from 0.625703 to 0.255138 eV. However, it still reverses the n-pi/pi-pi ordering in uracil and thymine. It is a useful experimental improvement, not a validated production solution to the original ordering problem.

29 October 2026. Parameter-free retrospective test, executed only on verified Oracle. Existing ground and excited-state wavefunctions were retained. No new SCF or excited-state optimization was launched.

`Omega_hybrid = Omega_DFT_frozen + (Omega_SE_relaxed - Omega_SE_frozen)`

Each frozen excitation is `2 E_mixed - E_triplet - E_ground`, evaluated on the respective unchanged canonical ground orbitals. DFT uses PBE0/aug-cc-pVDZ and a common, unpruned level-3 integration grid for all densities. SE uses the existing polarized GFN2 functional. There is no orbital-energy scaling, extent factor or fitted parameter.

10 states are qualified out of 11 planned. The uracil/thymine development cohort is reported separately from the previously tested six-state cohort; none is a new blind holdout for this newly proposed model.

## Mean absolute errors

| Cohort | States | SE ROKS / eV | Frozen DFT + SE relaxation / eV | Frozen DFT alone / eV |
|---|---:|---:|---:|---:|
| All qualified | 10 | 0.625703 | 0.255138 | 0.860018 |
| Uracil and thymine | 4 | 0.894927 | 0.320589 | 1.052584 |
| Previous generalization cohort | 6 | 0.446220 | 0.211504 | 0.731641 |

## Individual vertical energies

| Molecule | State | QUEST / eV | SE ROKS / eV | Hybrid / eV | Hybrid error / eV |
|---|---|---:|---:|---:|---:|
| nitroxyl | n_pi | 1.743 | 1.728 | 1.416 | -0.327 |
| uracil | n_pi | 4.935 | 4.319 | 5.164 | +0.229 |
| uracil | pi_pi | 5.381 | 4.276 | 5.024 | -0.357 |
| thymine | n_pi | 4.980 | 4.334 | 5.249 | +0.269 |
| thymine | pi_pi | 5.279 | 4.066 | 4.852 | -0.427 |
| nitrous_acid | n_pi | 3.527 | 3.054 | 3.496 | -0.031 |
| propynal | n_pi | 3.808 | 3.582 | 4.257 | +0.449 |
| pyridine | n_pi | 4.959 | 3.893 | 4.959 | +0.000 |
| pyridine | pi_pi | 5.133 | 4.827 | 5.440 | +0.307 |
| cytosine | pi_pi | 4.736 | 4.145 | 4.581 | -0.155 |
| cytosine | n_pi | 5.179 | — | excluded | — |

## Decomposition

| Molecule | State | SE frozen / eV | DFT frozen / eV | SE relaxation / eV |
|---|---|---:|---:|---:|
| nitroxyl | n_pi | 1.853392 | 1.540789 | -0.124923 |
| uracil | n_pi | 6.273673 | 7.118453 | -1.954226 |
| uracil | pi_pi | 4.695780 | 5.443767 | -0.419531 |
| thymine | n_pi | 6.023398 | 6.938853 | -1.689603 |
| thymine | pi_pi | 4.487651 | 5.273739 | -0.421852 |
| nitrous_acid | n_pi | 3.143521 | 3.585807 | -0.089776 |
| propynal | n_pi | 4.356635 | 5.031817 | -0.774943 |
| pyridine | n_pi | 4.988836 | 6.055267 | -1.095888 |
| pyridine | pi_pi | 5.579881 | 6.193112 | -0.752991 |
| cytosine | pi_pi | 5.048117 | 5.484633 | -0.903180 |

## State ordering

- uracil: pi-pi minus n-pi = -0.139991 eV; QUEST +0.446000 eV; correct ordering: False.
- thymine: pi-pi minus n-pi = -0.397364 eV; QUEST +0.299000 eV; correct ordering: False.
- pyridine: pi-pi minus n-pi = +0.480742 eV; QUEST +0.174000 eV; correct ordering: True.

These residual ordering errors cannot be attributed to the integration grid. The experiment does not by itself identify whether the remaining mismatch comes from the approximate frozen DFT excitation, the transfer of SE relaxation, or both. The decomposition makes a subsequent comparison of DFT and SE orbital relaxation a concrete diagnostic. No additional spectral scaling or fit was introduced to repair the ordering.

## Exclusions and checks

- cytosine n_pi: unqualified or ambiguous reference orbital mapping: [20].

State identities are the previously qualified common-orbital singlet solutions, with orbital-stationarity and CSF/transition-overlap checks. Ground orbitals are matched by actual cross-AO overlaps within separate occupied/virtual sectors (minimum 0.7 and margin 0.05). Energies are never selected by proximity to QUEST.

The maximum UKS-ground versus cached-SCF energy difference is 2.5067947717616335e-11 hartree. Total electron counts are checked in the AO metric; mixed and triplet densities have identical total density. Spin-flip invariance is checked for the first HNO and uracil states.
Independent level-4 grid check on HNO and uracil: maximum frozen-excitation change 0.000022884 eV, declared tolerance 0.005 eV; passed: True.

The motivation to separate frozen excitation and relaxation is related to [Ling et al., JCTC 2026](https://doi.org/10.1021/acs.jctc.5c01738). The present use of frozen canonical ground orbitals and combination across two different methods is an experimental hypothesis, not that paper's derived EDA or a proven correction.

This is an energy-only experiment. Nuclear derivatives of the DFT frozen block and the ground-orbital matching are not supplied; existing SE gradients/Hessians must not be relabeled as hybrid derivatives. The DFT setup and static exchange/XC evaluations contribute to the total computational cost, even though no excited-state SCF is needed. No new claim of equal-accuracy speed over TDA is made.

Ground, mixed and triplet energy points are individually cached with request hashes. Reproduction on Oracle uses its existing interpreter and OMP/BLAS thread limits of two:

```
python frozen_relaxation.py
python frozen_grid_check.py
python frozen_relaxation_summary.py
```

QUESTDB commit 460482bcc6231dab41fe32188dc43959a7c2bdaa, CC BY-SA 4.0; source metadata/license are retained in the preceding campaigns.
