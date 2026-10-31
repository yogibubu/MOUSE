# Frozen-profile generalization test

Numerical work on oracle.sns.it. Parameters fixed before this cohort: lambda0=0.870620576673, power=1.5; no refitting.

**The calibrated radius-ratio profile is not generally accurate.** It substantially overestimates the small-molecule n-pi excitations, despite strong state-identity overlaps. Its paired-cohort MAE is higher than that of the original semiempirical ROKS energies. The uracil/thymine results are therefore limited evidence within a related chemical family, not a general calibration.

Qualified 6 of 7 planned states on 5 new molecules. Uracil and thymine are excluded from these statistics.

| Molecule | State | QUEST / eV | New ROKS / eV | Error / eV | Previous error / eV | Index |
|---|---|---:|---:|---:|---:|---:|
| nitroxyl | n_pi | 1.743 | 4.008 | +2.265 | +2.560 | 0 |
| nitrous_acid | n_pi | 3.527 | 4.913 | +1.386 | +1.842 | 0 |
| propynal | n_pi | 3.808 | 6.515 | +2.707 | +2.553 | 0 |
| pyridine | n_pi | 4.959 | 5.998 | +1.039 | +1.359 | 0 |
| pyridine | pi_pi | 5.133 | 5.472 | +0.339 | +0.755 | 2 |
| cytosine | pi_pi | 4.736 | 5.120 | +0.384 | +0.682 | 2 |
| cytosine | n_pi | 5.179 | excluded | — | — | — |

## Coverage and accuracy

{
  "roks_calibrated": {
    "count": 6,
    "mae_ev": 1.3535641628465391,
    "mean_signed_error_ev": 1.3535641628465391,
    "maximum_absolute_error_ev": 2.707478830036537
  },
  "roks_previous_ratio": {
    "count": 6,
    "mae_ev": 1.6251212413953906,
    "mean_signed_error_ev": 1.6251212413953906,
    "maximum_absolute_error_ev": 2.559829414114848
  },
  "roks_original_se": {
    "count": 6,
    "mae_ev": 0.44621987389049905,
    "mean_signed_error_ev": -0.44621987389049905,
    "maximum_absolute_error_ev": 1.0660523698928461
  },
  "tda_calibrated": {
    "count": 6,
    "mae_ev": 1.1962366944463874,
    "mean_signed_error_ev": 1.1962366944463874,
    "maximum_absolute_error_ev": 2.335670991344703
  },
  "paired_roks_tda": {
    "count": 6,
    "roks": {
      "count": 6,
      "mae_ev": 1.3535641628465391,
      "mean_signed_error_ev": 1.3535641628465391,
      "maximum_absolute_error_ev": 2.707478830036537
    },
    "tda": {
      "count": 6,
      "mae_ev": 1.1962366944463874,
      "mean_signed_error_ev": 1.1962366944463874,
      "maximum_absolute_error_ev": 2.335670991344703
    }
  }
}

Statistics use identical accepted ROKS states for the calibrated, previous-ratio and original-SE models. Exclusions remain visible; no selection uses distance to QUEST energy.

The original SE ROKS n-pi energy of HNO is 1.728469 eV versus QUEST 1.743 eV. Applying the DFT spectrum replacement raises it to 4.007790 eV. This directly identifies the spectral correction as a source of error in this case, rather than a failure to locate the correct state. The radius ratio alone does not determine how to replace an SE orbital gap while retaining the SE interaction and relaxation corrections.

## Exclusions

- cytosine n_pi: spectral mapping: unqualified or ambiguous reference orbital mapping: [20] | no stationary state passed identity checks

## State identity and solver scope

Targets are linked to retained symmetry/character assignments and projected reference transition amplitudes, then checked against the relaxed GFN2 state. Canonical seed CSF overlap and reference transition overlap must both exceed 0.7; orbital chemical character must agree. Cross-basis orbital mapping uses overlap >=0.7 and margin >=0.05.

The initial protocol tries electronic Hessian indices 1–3. Additional index-zero searches are recorded separately after positive-index searches lost identity or failed. Index zero is a minimum of the restricted common-orbital two-determinant CSF functional, not a claim that the excited state is a minimum of an unrestricted full electronic functional. State identity thresholds and energy-model parameters were not changed. See [Kowalczyk et al.](https://pubmed.ncbi.nlm.nih.gov/23635105/) and the [Q-Chem ROKS manual](https://manual.q-chem.com/5.3/ex_ESROKS.html).

Full-functional GFN2 TDA uses the same GFN2 coefficients and the identical corrected reference spectrum. TDA values and their qualification/space limitations are retained per state in summary.json. The profile was calibrated to ROKS; these values do not establish an algorithm-neutral accuracy comparison.

All ground states, unconverged latest orbitals and converged states are checkpointed. DFT references were cached PBE0/aug-cc-pVDZ calculations at the same geometries; no new DFT SCF was run.

QUESTDB reference commit: 460482bcc6231dab41fe32188dc43959a7c2bdaa, CC BY-SA 4.0. Source metadata files and DFT references are hashed in each molecule result; local input copies retain the license.


## Checkpoint replay and reproduction

All six accepted states were reloaded and their spectral corrections replayed through the molecular API, with zero difference from the recorded energies. The saved states have S(S+1)=0, orbital residuals below 1e-9 and the stated Hessian indices. Energy-only checkpoints are loaded with explicit `require_gradient=False`; the default derivative loader still requires a finite gradient. The updated regression suite passes 36 tests on Oracle.

Run with the same remote interpreter and two-thread environment as the preceding campaigns:

```
python generalization.py
python generalization.py --index-zero
python generalization_summary.py
python generalization_audit.py
```

Both original positive-index attempts and additional index-zero attempts remain available, including rejected identities and failures. No model parameter was adjusted after observing this cohort.