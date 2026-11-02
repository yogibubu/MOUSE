# Chlorophyllide ROKS stationarity repair

All electronic work ran on the verified MAC-STUDIO. The native model and frozen quadrature are unchanged: F/Q 50/50, exchange lambda 0.9, bath eta 0.60, energy offset 0. No new SCC, TDA or DFT solve enters the prepared-model optimization.

Four distinct coherent singlets are certified. In initialization order their bare ROKS gaps are 2.849752, 2.715719, 3.341362 and 3.648886 eV. These are stationary basis states before screened CID, not final spectral predictions. The CSF metric eigenvalues are 0.688173, 0.836139, 1.066701 and 1.408987. The fourth checkpoint was already complete and remains byte-identical.

- `report.json` assembles preserved checkpoints, initialization histories and final certificates.
- `stationarity_audit.json` checks root identity and rejects the duplicate native target-3 solution.
- `final_revalidation.json` independently verifies the original full KKT residual at all four stored roots with zero optimization iterations, no retraction and no multiplier adjustment. The final source fingerprints match `qualified_sources/`.
- `controls_final/` contains 25 passing regressions, twelve molecular root-recovery controls, the ethylene analytic-Hessian regression and an independent production-checkpoint derivative audit.
- `before*/` preserves earlier checkpoints; `after/` holds the certified endpoints. Nonconverged searches and the rejected duplicate remain in `logs/`.
- `exploratory_sources/` retains a superseded globalizer generation. Its unused RFO paths are absent from the final production implementation.

The repair uses tangent-constrained SVD trust steps with exact multiplier elimination, checkpointed trust radii, and direct-energy initialization. A fixed-open-orbital relaxation precedes full release for the higher targets. The third root uses a hole orthogonal to the known root within the unchanged native occupied frontier block; that restriction applies only to its initial guess. Final certification uses every nonredundant orbital rotation on the original functional.

The corresponding scripts are `run_pigment_energy_escape.py`, `finalize_pigment_roks.py`, `revalidate_pigment_roks.py` and `collect_roks_stagnation_repair.py` in the parent qualification folder. Completed target certificates are checked before loading the large model. The finalizer automatically adopts only independent certified roots and writes the production `result.json`.

The large prepared reference arrays remain on MAC-STUDIO; their manifests and checksums are archived here. This evidence certifies vertical energies and orbital stationarity. It does not certify pigment nuclear gradients/Hessians or a screened-CID spectrum.
