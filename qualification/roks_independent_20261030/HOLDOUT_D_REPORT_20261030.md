# Frozen 50/50 independent cohort D

Three molecules fixed before their predictions: acrolein, pyrimidine, nitrosomethane. All three state assignments qualify. No fitted energy offsets or post-test weight changes. The previously fitted weight diagnostics remain unused.

| Molecule | TBE | Native ROKS | Complete response | Spectral response | Fixed 50/50 |
|---|---:|---:|---:|---:|---:|
| acrolein | 3.7230 | 3.1981 | 3.9762 | 3.5654 | 3.7708 |
| pyrimidine | 4.4510 | 3.2299 | 4.0856 | 4.3498 | 4.2177 |
| nitrosomethane | 1.9520 | 2.0636 | 1.6147 | 1.8277 | 1.7212 |

Energies in eV. MAE: native=0.619169, raw_full=0.318641, spectral=0.127708, energy_mean_50_50=0.170636, tda_arithmetic_control=1.270104.

**Original accuracy gate: FAILED.** The 50/50 mean beats native and complete-response ROKS, but does not beat the spectral component. This does not justify adjusting the weight on D.

TDA controls use each corrected GFN2 ground-state orbital basis and its own coherent kernel. Their arithmetic mean is a numerical composite control, not TDA of a single variational functional. Preparation costs are distinct from electronic solver timings. These vertical nonlocal models have no implemented nuclear gradients or nuclear Hessian. Existing shell/tensor analytic nuclear Hessians do not qualify these models.

See results/holdout_D_summary.json, ensemble_holdout_D_protocol.json and results/ensemble_weight_decision.json for the frozen protocol and full identities.
