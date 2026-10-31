# DFT reference gaps with semiempirical TDA and ROKS

Experimental qualification, 29 October 2026. All electronic calculations and
matrix diagonalizations ran on verified `oracle.sns.it`; no laptop electronic
jobs were run. Existing DFT and GFN2 wavefunctions were resumed from saved files.
No new SCF jobs were necessary for that constant-scaling pilot. The subsequent
extent study reuses these references and adds a checkpointed GFN2 ground
reference and two ROKS states for thymine; its DFT references were already saved.
See [the extent-scaling update](EXTENT_REPORT.md) for the current results.

## Shared orbital-energy scaling

The selected comparison uses the user-requested uracil value **lambda = 0.9**
on the reference orbital energies, identically for both methods:

```
delta_gap(i,a) = lambda * (eps_DFT[a] - eps_DFT[i])
                         - (eps_SE[a] - eps_SE[i])
Omega_ROKS_hybrid = Omega_ROKS_SE + delta_gap(hole,particle)
A_TDA_hybrid = A_TDA_SE + diag(delta_gap(i,a))
```

One immutable `MatchedOrbitalSpectrum.reference_energy_scale` supplies both
corrections. The SE orbitals, kernel, spin polarization and relaxation energy
are unchanged. TDA is rediagonalized after correcting configuration diagonals;
root energies are not simply shifted by a selected dominant promotion.
The original SE gap is subtracted without multiplying it by lambda.

The preexisting imported-DFT TDA protocol applies its lambda = 0.9 to the
response exchange fraction (`a_response = 0.9 * a_ground`), and does not scale
orbital energies. Applying 0.9 to reference gaps here is the explicitly
requested additional model prescription. These are separate operations;
the current comparison retains each cached SE kernel without an extra kernel
rescaling. Earlier unscaled pilot outputs remain in `results/` for provenance;
the selected results are exclusively in `results/scale_0.9/`.

## Orbitals and configuration spaces

Both response kernels use **GFN2 coefficients**, not DFT coefficients. DFT
coefficients are used only to match the reference spectrum by actual cross-AO
overlaps, independently in occupied and virtual sectors. Nuclear geometry,
elements, AO metrics and orthonormality are checked. Matching is independent
of orbital energies; the DFT n/pi ordering differs from GFN2.

For uracil at the exact QUEST geometry, n-hole 20 and pi-hole 19 map to DFT
orbitals 27 and 28; particle 21 maps to DFT orbital 29 (zero-based indices).
B3LYP orbital overlaps are approximately 0.954, 0.973 and 0.934.
The overlap >= 0.7 and margin >= 0.05 criteria retain two virtual orbitals
and 21 occupied orbitals, giving 42 configurations out of the full 315.
Low-overlap higher virtuals are excluded from the primary comparison.

## Vertical energies

All entries are eV at the same fixed geometry. ROKS denotes the common-orbital
spin-pure two-determinant singlet model with polarized GFN2, located by orbital
saddle search. Its two target solutions have electronic Hessian indices 1 and 2.
The main TDA column is the singlet TDA of the same GFN2 energy functional;
it is distinct from the empirical compact sTDA kernel.

| DFT reference, lambda=0.9 | ROKS n-pi | ROKS pi-pi | TDA n-pi, 42 CSFs | TDA pi-pi, 42 CSFs |
|---|---:|---:|---:|---:|
| B3LYP/aug-cc-pVDZ | 5.966700 | 5.463868 | 5.375439 | 6.434632 |
| PBE0/aug-cc-pVDZ | 6.415160 | 5.850489 | 5.823904 | 6.852592 |
| PBE0/aug-cc-pVTZ | 6.418482 | 5.856282 | 5.827226 | 6.858359 |
| QUEST correlated TBE | 4.935 | 5.381 | 4.935 | 5.381 |

Uncorrected SE ROKS gives 4.319446 and 4.276250 eV. Uncorrected functional TDA
gives 3.728082 and 5.098643 eV in the same 42-configuration space, or 3.727761
and 4.827933 eV in the full space. Thus the pi-pi truncation effect is 0.270710 eV;
it must not be attributed to the reference-spectrum correction.

B3LYP hybrid ROKS improves pi-pi substantially (error +0.083 eV), while n-pi
is too high by 1.032 eV and the ordering remains reversed. Two states do not
establish transferable accuracy or justify replacing TDA in production.
PBE0 basis enlargement changes these ROKS energies by less than 0.006 eV,
but the functional dependence is substantially larger.

The independent empirical compact sTDA kernel yields 0.513380 and 1.687287 eV
with the same B3LYP spectral correction and 42-configuration space. Its poor
absolute energies persist; this kernel must not be confused with functional TDA.

## Space sensitivity and derivatives

For B3LYP, lowering the orbital-overlap threshold from 0.7 to 0.6 to 0.5 retains
42, 63 and 84 configurations. Functional TDA n-pi becomes 5.375439, 5.375365,
5.375341 eV; pi-pi becomes 6.434632, 6.259042, 6.248340 eV. Those expanded spaces
use less reliable orbital correspondences and are diagnostics, not an improved
qualified mapping chosen after seeing the energies. The primary pi-pi TDA
comparison remains sensitive to the virtual-space restriction.

The correction is independent of the excited-state orbital rotations at fixed
geometry, so it preserves the SE orbital stationary point and Hessian index.
It changes the nuclear surface. Hybrid gradients and analytic Hessians require
first and second nuclear derivatives of **lambda*gap_DFT - gap_SE**. The module
combines explicit correction derivative jets and rejects requests that would
silently reuse SE derivatives. A molecular DFT spectral-response implementation
is not supplied by this pilot; the existing analytic SE ROKS Hessian is not a
complete analytic Hessian of the hybrid surface.

There is no new equal-accuracy timing benchmark here. Reusing existing DFT
references avoids new SCF jobs for this pilot, but generating reference spectra
and their nuclear derivatives must be included in any end-to-end cost comparison.

## Reproduction and verification

On the verified Oracle host, in the matching qualification directory:

```
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/enzo/nano-MATRIX-current/.venv/bin/python compare.py --reference-energy-scale 0.9
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/enzo/nano-MATRIX-current/.venv/bin/python sensitivity.py --reference-energy-scale 0.9
```

The remote bootstrap verifies the host, uses the isolated source snapshot and
reuses the installed Linux native extension. `provenance.json` records hashes
of source, scripts, DFT checkpoints and results. `tests_scaling.log` records
**31 passing tests**, including identical scaling in TDA/ROKS, gauge invariance,
unchanged couplings, scaled analytic derivative composition and derivative guards.
The molecular spectra are in `results/scale_0.9/comparison.json`; raw eigenpairs,
mapped reference spectra and overlap-threshold sensitivity are retained alongside.

QUEST reference provenance and license are retained in the preceding
`../uracil_uridine_20261029/` campaign: QUESTDB commit
`460482bcc6231dab41fe32188dc43959a7c2bdaa`, CC BY-SA 4.0.
