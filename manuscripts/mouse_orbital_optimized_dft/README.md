# MOUSE semiempirical OO-DFT manuscript

Scientific draft in English, 30 October 2026. Validation status updated 1 November 2026.

The final release uses the corrected nonlocal singlet TDA operator on stationary
corrected GFN2 orbitals. Archived static-response matrices are superseded and
excluded from accuracy and timing comparisons. The final frozen primary cohort,
full-anchor recalibrations, physical-reference checks and six-size cost ledger
are packaged in `data/final_validation.json`; figures and tables rebuild using
`python3 make_final_validation_assets.py` without electronic calculations.

The central contribution is a semiempirical orbital-optimized density-functional-response-informed method with analytic excited-state gradients and nuclear Hessians. TDA on corrected GFN2 orbitals and a separate published dedicated sTDA-xTB calculation are the comparison methods. The implemented candidate is the fixed 50/50 relative-active-pair constrained ROKS composite, not a full-basis DFT calculation at each geometry.

## Read

- `main.pdf`: complete manuscript.
- `supporting_information.pdf`: all vertical results, reference methods, derivative checks, optimized coordinates and frequencies, integration settings.
- `main.tex`, `supporting_information.tex`, `references.bib`: editable sources.
- `data/vertical_energies.csv`: 19 states and two TDA controls, with exposure labels.
- `data/SUMMARY.json`: measured results; `data/PROTOCOL.json`: validated calculation source hashes.
- `reference_provenance.json`, `data/tbe_provenance.json`: inherited reference metadata and state-specific TBE methods.
- `coordinates/`: molecular minima and reference anchors in XYZ format.
- `literature_metadata.json`: retrieved Crossref records. Unavailable records have explicit errors; the corresponding BibTeX entries were checked against publisher or author-hosted records.

## Scientific framing

Levi and co-workers motivate direct orbital optimization and electronic saddle searches (2020, 2021, 2023, 2026; DOI links in the bibliography). Their published results are not attributed to MOUSE. The manuscript distinguishes a spin-pure CSF from an exact projected DFT energy, and a constrained composite stationary surface from an unconstrained single density functional.

The main validation figures are 0.183 eV MAE for the curated 19-state cohort and 0.304 eV for the four prospective molecular tests. The previously quoted 1.029/1.073 eV static-response controls are withdrawn. The corrected 14-state TDA-Q MAE is 0.261 eV; the final primary frozen cohort accepts four of six states with OO MAE 0.675 eV. These four molecules had earlier predictions with other models. The text makes no claim of a wholly blind benchmark, universal n-to-pi-star correction, experimental-frequency accuracy, or faster energy-only execution.

## Rebuild

From this folder:

```sh
python3 build_assets.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error supporting_information.tex
pdflatex -interaction=nonstopmode -halt-on-error supporting_information.tex
```

`build_assets.py` reads the existing qualification archive in the associated MOUSE working tree and performs data aggregation and plotting only. No electronic calculations are launched. Existing electronic results were computed remotely on Oracle. The author and affiliation follow the repository's CITATION.cff; the draft has no assigned target journal or permanent dataset DOI.

## Positioning revision

The revised introduction and discussion explicitly distinguish sTDA/sTD-DFT, dedicated sTDA-xTB, Xs variants/std2 (2026), TD-DFTB and recent hybrid parameterizations, TDDFT+TB/TDDFT-aas, Delta-DFTB, and OMx/ODMx-MRCI. A comparison table separates the electronic reference, excited-state formulation, and documented derivative capabilities. Thirteen additional primary references are stored in `positioning_references.json`.

The justification for semiempirical treatment concerns repeated molecular structures, orbital relaxation, and derivative-based work. Reference preparation and dense AO-pair response costs remain explicit; the manuscript does not claim an established total-cost advantage over DFT or a transferable accuracy ranking against published sTDA-xTB. The direct 19-state comparison gives MAEs of 0.183 eV (MOUSE) and 0.454 eV (sTDA-xTB), while the four previous prospective cases give 0.304 and 0.278 eV, respectively.

## Publication figures

Six figures use a consistent vector style and are supplied in PDF, SVG and high-resolution PNG: `method_workflow`, `stationary_response`, `vertical_validation`, `molecular_minima`, `hessian_validation`, and `direct_stda_validation`. Run `python3 make_publication_figures.py` to redraw them from the copied data and XYZ files; `build_assets.py` also calls this final styling step. `figures/FIGURE_PROVENANCE.json` distinguishes actual measured results and geometries from the explicitly labelled quadratic saddle illustration. No new electronic jobs are needed to reproduce the figures.

## Publication-validation extension

`extend_publication_results.py` generates the direct released sTDA-xTB comparison, all-state external-method table, structural comparison and vertical-anchor weight precheck. It is called by `build_assets.py`. Completed external results are copied into `data/`; `direct_stda_comparison.csv` preserves every root, overlap, projected norm and margin. The 50/50 candidate is unchanged.

The original 44-state manifest and all dispositions are retained. The active scope was reduced after baseline convergence inspection to 14 states on ten molecules. All 14 states pass stationarity and available-root audits; matched TDA controls and prediction closure are complete. Reference labels were opened only after the lock. OO MAE is 0.492 eV versus 0.261 eV for TDA-Q; the single-CSF pi-pi error is substantial. This reduced/adapted set is not wholly blind. The external DFTB symmetry audit and the final full-anchor, derivative and memory campaign are reported in later sections; a full lambda scan remains outside this study.

Internal-coordinate results against CCSDR(3)/def2-TZVPP are available for three minima. Nitroxyl harmonic-frequency comparisons remain provisional by frequency order until mode-vector confirmation. These physical comparisons must be kept separate from numerical validation of the analytic derivatives.

## Revision of 31 October 2026

Code commit `5620e2d` includes spin-pure ROKS/DeltaSCF, analytic nuclear response, APOC/LINK integration, batched orbital response and assigned-NTO initialization. Main Sections 2.6--2.7 and 4.4 and SI Section 8 explain the solver, amended identity policy and measured timing scope. Seven scientific figures use archived results or labelled schematics. The obsolete static-response accuracy and timing comparisons have been withdrawn as TDA results.

## Molecular-structure plate

`figures/molecular_structures_jmol.pdf` shows all 28 molecules in two separate cohorts (18 initial, 10 extension), using actual Jmol ball-and-stick renders of the archived calculation anchors. The extension includes the completed phthalimide targets; presence in the plate is not a convergence or accuracy claim. `data/molecular_structures_jmol.json` records geometry checksums, proper display rotations and Jmol scripts; `data/structure_geometry_sources.json` records the Oracle geometry sources and hashes. All ten transferred extension files match the frozen geometry hashes where present. No optimization is performed.

Rebuild with `python make_structures_jmol.py --jmol-jar /path/to/Jmol.jar` (Jmol 14.31.10 used). Render caches are optional and stored under `tmp/jmol/`; PDF assembly uses ReportLab and Pillow, orientation uses NumPy.

## Completed extension and multi-CSF development

`build_reduced_extension.py` regenerates the complete locked 14-state tables and a separately labelled exposed-data experiment. The 95%-NTO-rank bare-coupling trial improves the five pi-pi states (MAE 0.162 eV) but severely worsens the aza-naphthalene n-pi control; it is not a universal correction or a production method. Nuclear qualification of this new energy is separate from the existing single-CSF analytic Hessian. Commit `49e0c9d` accelerates resolved soft saddle modes, with 31 remote regressions passing. No additional molecules have been added.

The full-frame nuclear response now includes common/private product-chart and spectator-gauge mixed curvature. Two Hessian columns pass at the Furan anchor for the complete excited total energy and at a displaced geometry for the constrained parent mean, within 2.3e-8 Ha/bohr^2. This does not certify every molecular column or general transferability. Figure 9 compares all 14 state errors and family MAEs.

## Public multi-CSF integration (489e899)

The explicit `multicsf_50_50` selector now supports portable frozen models,
coherent APOC expansions, direct LINK scans/optimization, analytic Hessians and
atomic accepted-state restart. All 27 furan Hessian columns pass within
2.2852e-8 Ha/bohr². Forty-three core/legacy/APOC tests and three further LINK
contract tests pass. Cold full-Hessian evaluation is 98.99 s; restoring the
accepted state and its stored Hessian is 0.613 s, not a new-Hessian benchmark.

Figure 10 and `make_integration_assets.py` report response-only lambda,
algebraic vertical-weight and fixed-reference rank sensitivity. The final
DFT-anchor recalibration, independent validation, physical-frequency comparison
and six-size cost/memory assessment are reported in the final campaign. The 50/50 weight and lambda
0.9 remain fixed. Phthalimide's rank boundary is a material model limitation.

## External DFTB symmetry audit

A read-only audit of completed Oracle outputs supplies TD-DFTB2/mio-1-1
symmetry/ordinal comparisons for all original 19 states: MAE 0.54547 eV versus
0.18253 eV for MOUSE. Fifteen completed, eligible lowest-singlet Delta-DFTB
cases have the correct final determinant spatial irrep: MAE 0.31353 eV versus
0.17174 eV for MOUSE on precisely those states. Two incomplete SCC attempts
and two higher reference states remain explicit. This is a weaker identity
certificate than cross-model transition-overlap qualification; roots sharing
an irrep may differ in character. It is not new independent validation.

`build_dftb_audit.py` rebuilds Supporting Table S7 from standalone copied JSON.
The qualification archive contains 254 hash-verified input/result artifacts,
including actual DFT reference wavefunctions and DFTB eigenvectors, transition
weights, occupations and energies. No electronic calculation was repeated for
this audit. The energy model and all weights/scalings remain unchanged.

## Complete second-molecule derivative validation

All 33 cyclopentadiene excited-total Hessian columns pass through the public
multi-CSF interface, maximum error 1.2964e-7 Ha/bohr². Along with all 27 furan
columns, Figure 11 now shows every measured column and independent
energy/gradient difference. Both stored molecular Hessians pass translation and
nonstationary rotation covariance checks. These are derivative-consistency
checks, separately from physical frequency accuracy.

The cold cyclopentadiene public-surface Hessian call is 250.27 s and 858 MiB
peak process RSS after model loading. Its 66 independently reoptimized signed
energy/gradient calls total 465.47 s (median 7.05 s), a 1.86 ratio to the analytic
call for this prepared model. Neither includes DFT/reference preparation, and
this is not a TDA or a size-scaling comparison. `make_second_hessian_assets.py`
rebuilds Figure 11 and Supporting Table S8 from copied standalone JSON.

## Actual reference/rank sensitivity

Figure 12 and Supporting Table S9 show all six recomputed DFT/native-S0
projection points at +/-2% internal Cartesian scale. Every point passes the
fixed transition-identity and projection thresholds and retains rank two.
The final exact-irrep response solves certify both roots against the full
operator at the unchanged 1e-7 residual threshold. Earlier uncertified restart
vectors are retained in the archive and enter no accepted result.

This closes the specified local rank diagnostic, with all 185 new artifacts
hash verified. It does not close full F/Q recalibration with energies evaluated
at a common target geometry, independent molecular validation, physical
structure/frequency accuracy, or a complete TDA/OO workflow cost and size study.
`make_anchor_rank_assets.py` rebuilds the plot and table from standalone JSON.

## Final campaign

All six full F/Q anchor tests are complete. The primary, new-molecule and
higher-state cohorts stay separate, with locked predictions and full coverage
records. A symmetry-breaking cyclopentadiene minimum has 27 positive frequencies;
Furan is compared to the corresponding symmetric EOM-CCSD stationary saddle.
The six-size benchmark reports prepared E, E+G, E+G+H and true GFN2-orbital TDA,
with process memory and separately audited preparation stages. Missing historical
cold-preparation totals remain unknown. The candidate is not a universal accuracy
or speed improvement over TDA. Source, checkpoints, bounded diagnostics and
qualification records are in `qualification/final_validation_20261031`.

The final analytic-response contraction completes Phthalimide's full 48-by-48
Hessian in 42.224 s on Oracle with two threads, versus a previous lower bound
of 600 s. All six benchmark Hessians are complete; affected derivative costs
are remeasured. The benchmark peak RSS falls to 1.07 GiB. Literal-jet tests,
whole-matrix comparisons, controlled gradient differences, and 50 regression/
contract/response tests qualify the unchanged mathematical model. The exact
implementation is committed as `4fe3cd8`.


The 1 November frontier-CISD pilot is documented separately in the SI. Its
eleven-state MAE is 0.217 eV; the prospective-only MAE is 0.385 eV and glyoxal's
0.552 eV discrepancy exceeds the fixed maximum-error gate. The pilot is not
promoted as a universal correction. It has independent analytic molecular
Hessian checks for Criegee and Furofuran and TDA-free ground-program recovery
for all four corresponding ROKS states. Code/evidence: 5f69635; candidate
freeze: 7666f7a. `make_frontier_cisd_assets.py` reads only the bundled
`data/frontier_cisd` files, so its figure and table rebuild independently of
the remote electronic calculations.

The final selected optimized-target CID candidate is frozen as 467a9e5 and
implemented in the public MOUSE/APOC/LINK interfaces in ecb6644. All fifteen
valence transitions meet the fixed gates: MAE 0.219905 eV, maximum 0.497948 eV.
Thirteen informed model selection; two post-freeze predictions on historically
exposed molecules give 0.201994 eV MAE. No universal or blind claim is made.
Three complete molecular analytic Hessians cover five states. The public
Criegee Hessian reproduces the qualified matrix within 9.1e-13 Hartree/bohr²;
actual LINK cold-start/rollback/restart and coherent spin-pure following pass.
The SI preserves the earlier unsuccessful pilots separately. Rebuild the new
figure and table with `python make_selected_cid_assets.py`; all its inputs are
bundled in `data/selected_cid`.
