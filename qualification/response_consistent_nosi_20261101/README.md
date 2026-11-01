# Selected optimized-target CID qualification, 1 November 2026

Final candidate: optimized ground-orthogonal target ROKS singlets and their
paired doubles, frontier canonical double excitations and exact four-spin
singlets. No unoptimized auxiliary canonical singles; no screened or Pauli-fit
operator modification. Moving stationary-ground projector. Common F/Q one-body
reference with weights 0.5/0.5, exchange response lambda 0.9, zero fitted offset.

Freeze: `467a9e5`, before nitrosomethane and maleimide predictions. Thirteen
states are exposed development; two post-freeze states belong to molecules
historically used by older models. This is not a blind benchmark. The fixed
MAE <= 0.3 eV and maximum <= 0.5 eV gates pass for all 15 states: MAE
0.21990523 eV, maximum 0.49794804 eV (glyoxal). The latter is close to the gate.
The post-freeze two-state MAE is 0.20199400 eV. `score_selected.py` rebuilds
`results/predictions.csv` and `results/summary.json` from raw outputs and the
unchanged reference values. Same-reference TDA uses identical native GFN2
spatial orbitals, not full-basis DFT orbitals. Its 15-state MAE is 0.96637685
eV; this does not represent the best achievable calibrated TDA model.

## Public surface

`matrix_gfn2.PreparedNOSIModel(path)` loads three checksummed numeric files.
`model.evaluate(calculator, coordinates_angstrom, properties=('energy',
'gradient','hessian'))` returns Hartree, Hartree/bohr and Hartree/bohr².
`model.apoc_states(result)` expands coherent determinant sums and verifies
S² = 0. The public LINK selector is `nosi_50_50`, with
`native_nosi_model_path` pointing to the model manifest. Only accepted
geometries advance the atomic restart. Model mutation is rejected even before
optimizer cache hits. Gas-phase closed-shell references only are qualified.

`results/prepared_model_criegee` is a complete portable example. It needs the
native MOUSE kernel, NumPy/SciPy and the repository packages; it needs no PySCF,
TDA amplitudes or DFT input at runtime. Preparation still requires a one-time
DFT reference; its cost must not be hidden. `export_selected_model.py` is the
adapter for the archived research preparation pipeline, not a general
Gaussian/ORCA input parser.

## Analytic derivatives and remote-only compute

Complete final-model Hessians are preserved for Criegee (2 states, 15
coordinates), glyoxal (1 state, 18 coordinates), furofuran (2 states, 36
coordinates). `results/selected_cid_analytic` contains full matrices, raw
symmetry/translation certificates and independent reconverged energy checks.
The analytic construction uses no displaced electronic calculations. Validation
uses two directions and two step sizes, with Richardson extrapolation to check
central-difference truncation. All implicit first/second orbital KKT response,
moving projection, overlap metric, and CI eigenvector response are retained.
Operators and auxiliary orbitals are anchored in the co-rotating orthonormal
AO frame; a fully re-canonicalized surface is not claimed.

Fresh-process audit: public Criegee Hessian agrees within 9.1e-13 Hartree/bohr²,
40.8242 seconds, 274776064 bytes peak RSS on MAC-STUDIO, 2 BLAS threads.
Energy-only: 0.1346 seconds. Preparation excluded. No DFT imports, no new SCC
calls, no TDA reads. `validate_link.py` verifies cold start, actual AO-overlap
coherent tracking, trial rollback and accepted restart. `validate_portable.py`
blocks PySCF imports and SCC calls. `test_nosi_contracts.py` and APOC unit tests
check identity, tampering, interference, spin purity and AO covariance.

All electronic calculations ran on verified Oracle or MAC-STUDIO. The laptop
only edited, transferred, inspected evidence and built lightweight figures/TeX.
Resume scripts preserve completed Hessian columns. MAC-STUDIO ru_maxrss is in
bytes, not KiB. Parallel furofuran timing excludes the earlier sequential
column and cannot be treated as a full serial scaling point. The old
phthalimide Hessian is a different model and is not relabeled as selected CID.

## Exposure and closed diagnostics

`remote_artifacts` retains all completed diagnostic results, checkpoints and
logs, including full-kernel, normal-ordered, Pauli-projected, spin-preserving,
atomic-spin, screened, kernel-only and coherent-diagonal variants. None is
silently pooled with the final selected CID predictions. Large operators remain
remote with hashes in the transfer audit; completed work was not rerun.
There are no claims of universal n→pi* correction, broad CT/Rydberg coverage,
blind generalization, or final-model coupled-cluster geometry/frequency accuracy.
The final candidate passes the specified finite valence-state test.
