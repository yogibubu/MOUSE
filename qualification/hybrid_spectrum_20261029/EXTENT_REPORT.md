# Orbital-extent reference-spectrum correction

This records the original unfitted power=1 hypothesis. The subsequent
[one-scalar calibration](CALIBRATION_REPORT.md) further reduces held-out errors
and uses an explicit power=1.5 profile.
That calibrated profile then fails the broader
[five-molecule generalization test](generalization/REPORT.md).

29 October 2026. Experimental development on uracil and independent check on
thymine. All molecular jobs, integral evaluations and matrix diagonalizations
ran on verified `oracle.sns.it`. Existing DFT checkpoints were reused; the
new thymine GFN2 ground and excited states were saved and resumed.

## Model and implementation

A particle-only lambda cannot separate n-pi and pi-pi here: both promotions
end in the same lowest pi-star orbital. The tested descriptor includes the hole.
Use the translation-invariant central spread of each canonical ground orbital:

```
sigma_p = sqrt(<p|r^2|p> - |<p|r|p>|^2)
w_i = sigma_SE[i] / sigma_DFT[matched_i]
epsilon_anchor = lowest reliably mapped DFT virtual energy
epsilon_effective[i] = epsilon_anchor + w_i*(epsilon_DFT[i]-epsilon_anchor)  (occupied)
epsilon_effective[a] = epsilon_DFT[a]                                     (virtual)
delta_gap(i,a) = 0.9*(epsilon_effective[a]-epsilon_effective[i])
                    - (epsilon_SE[a]-epsilon_SE[i])
```

For transitions into the anchor orbital, the effective gap multiplier is
`lambda_i = 0.9*w_i`. Higher virtuals use the same transformed spectrum rather
than independently rescaling each excitation gap. This preserves energy-gap
additivity and invariance to a common change of energy origin. The smooth ratio
has no clipping or fitted exponent. The anchor must be isolated and reliably
matched; the implementation rejects an ambiguous or degenerate anchor.

`OccupiedExtentScaling` retains immutable matched radii.
`with_occupied_extent_scaling` attaches the explicit hypothesis to an existing
`MatchedOrbitalSpectrum`. Both `correct_roks_excitation` and
`replace_tda_reference_gaps` use its `effective_reference_energies_hartree`.
The latter modifies configuration diagonals and rediagonalizes the matrix,
preserving the original SE response kernel and couplings.

No coefficient was fitted to QUEST energies. This is a new empirical model
hypothesis motivated by the mismatch between SE and DFT orbital sizes, not a
derived self-interaction correction. Uracil is the development system; the
unchanged law was then applied to thymine. Alternative hole/particle radius
powers are retained as exploratory diagnostics, not calibrated parameters.

Orbital-dependent corrections have precedents, including
[Vydrov and Scuseria's orbital-density scaling](https://pubmed.ncbi.nlm.nih.gov/16729796/)
and [Ivanov, Levi, Jonsson and Jonsson's direct optimization of orbital-dependent excited-state functionals](https://pubmed.ncbi.nlm.nih.gov/34227810/).
Neither paper establishes the radius-ratio formula proposed here.

## Uracil

At the same fixed QUEST geometry, the GFN2 central radii of n and pi are
3.046815 and 3.600117 bohr; matched B3LYP/aug-cc-pVDZ radii are 3.608487 and
3.763515 bohr. Therefore lambda_n = 0.759912 and lambda_pi = 0.860925,
despite the same starting global lambda of 0.9.

ROKS vertical gaps in eV:

| DFT reference | Constant lambda: n-pi | Extent: n-pi | Constant lambda: pi-pi | Extent: pi-pi |
|---|---:|---:|---:|---:|
| B3LYP/aug-cc-pVDZ | 5.966700 | 5.128822 | 5.463868 | 5.243912 |
| PBE0/aug-cc-pVDZ | 6.415160 | 5.505476 | 5.850489 | 5.633638 |
| PBE0/aug-cc-pVTZ | 6.418482 | 5.522827 | 5.856282 | 5.646317 |
| QUEST TBE | 4.935 | 4.935 | 5.381 | 5.381 |

The ordering becomes correct for all three references. With B3LYP the n-pi
error falls from +1.031700 to +0.193822 eV; pi-pi changes from +0.082868 to
-0.137088 eV. The splitting is still underestimated (0.115090 versus
0.446 eV), so this is an improvement, not complete quantitative agreement.
PBE0 retains appreciable absolute errors despite the corrected ordering.

The identical extent-transformed B3LYP spectrum in **functional GFN2 TDA**
gives 4.537564 and 6.125901 eV in the primary 42-configuration space.
Both methods use GFN2 coefficients. This space excludes poorly matched virtuals;
the full GFN2 rectangle has 315 configurations. TDA pi-pi retains significant
configuration-space sensitivity. The distinct empirical compact sTDA kernel
continues to give poor absolute energies and is not used as a qualified benchmark.

## Independent thymine check

The same PBE0/aug-cc-pVDZ prescription gives:

| State | Constant lambda | Extent scaling | QUEST TBE |
|---|---:|---:|---:|
| n-pi | 6.425935 | 5.447407 | 4.980 |
| pi-pi | 5.636340 | 5.560449 | 5.279 |

The n-pi error falls from +1.445935 to +0.467407 eV; pi-pi falls from
+0.357340 to +0.281449 eV. The ordering becomes correct. Changing to
aug-cc-pVTZ gives 5.465826 and 5.574198 eV. This supports transfer of the
direction of the correction to a related molecule, but does not establish
general accuracy across chemical families.

The retained thymine n-pi and pi-pi solutions have orbital Hessian indices
1 and 2, residuals below 6e-10, and the expected oxygen-in-plane/pi characters.
Canonical-hole overlaps exceed 0.98. Full singlet-CSF overlaps against the
corresponding canonical seeds are also checked with a 0.7 threshold.
The independent TDA values are retained in the molecular JSON; its truncated
space and state overlaps must accompany any comparison.

## Derivatives, scope and reproducibility

The correction depends only on the ground reference at a fixed geometry, so
the existing ROKS orbital stationary points and saddle indices are preserved.
Its nuclear derivatives require the radius ratios, DFT spectral response and
anchor response, including their cross terms. This study does **not** implement
these molecular analytic derivatives. Existing SE Hessians remain SE Hessians;
the hybrid API rejects a Hessian request without an explicit correction jet.
The smooth radius law permits differentiating an isolated, continuously tracked
orbital branch; orbital degeneracies require a subspace treatment.

No new equal-accuracy end-to-end timing claim is made. The molecular evidence
consists of two states on two related molecules; production defaults are unchanged.

On Oracle, using its existing validation environment:

```
python extent_probe.py
python compare.py --reference-energy-scale 0.9 --extent-policy basis_compactness
python extent_holdout.py
python run_tests.py
```

Use the interpreter and thread limits from the companion REPORT.md. The
`remote_bootstrap` module checks the host before importing the electronic code.
Outputs are retained under `results/extent/` and
`results/scale_0.9_occupied_extent/`. Ground, latest and converged checkpoints
allow resuming instead of restarting. An initial thymine driver metadata error
was repaired; the saved orbitals were resumed and the error record retained.
`provenance.json` records source, scripts, inputs and artifact hashes.

The remote suite has 34 passing tests, including common TDA/ROKS spectral
corrections, gauge invariance, unchanged couplings, invalid radii, ambiguous
anchors and nuclear-derivative guards. Exact AO moments reproduce the GFN2
overlap metric and were verified invariant to a change of moment origin.
QUEST source and license are the same as in the companion REPORT.md.
