# MOUSE semiempirical OO-DFT manuscript

Scientific draft in English, 30 October 2026. Validation status updated 31 October 2026.

The full-response GFN2 TDA controls in the current PDF and generated figures predate a confirmed nonlocal-exchange correction. They are archival static-response diagnostics, not qualified true-TDA comparisons. Their accuracy and timing comparisons must be regenerated before manuscript use. An independent PySCF A-matrix check verifies the correction to 3.6e-15 Hartree for exchange fractions 1 and 0.225. The OO-DFT energies, derivative checks and separate published sTDA-xTB results are unaffected. The corrected 14-state/10-molecule campaign is recorded in `../../qualification/state_following_20261031/reduced_validation`.

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

The main validation figures are 0.183 eV MAE for the curated 19-state cohort and 0.304 eV for the four prospective molecular tests. The previously quoted 1.029/1.073 eV static-response controls are withdrawn as TDA comparisons pending the corrected calculation. These four molecules had earlier predictions with other models. The text makes no claim of a wholly blind benchmark, universal n-to-pi-star correction, experimental-frequency accuracy, or faster energy-only execution.

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

The original 44-state manifest and all dispositions are retained. The active scope was reduced after baseline convergence inspection to 14 states on ten molecules. All 14 states pass stationarity and available-root audits; matched TDA controls and prediction closure are complete. Reference labels were opened only after the lock. OO MAE is 0.492 eV versus 0.261 eV for TDA-Q; the single-CSF pi-pi error is substantial. This reduced/adapted set is not wholly blind. TD-DFTB/DeltaDFTB state assignments, full lambda/anchor sensitivity and complete derivative/memory costs remain unfinished.

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
algebraic vertical-weight and fixed-reference rank sensitivity. The actual
DFT-anchor recalibration, independent validation, physical frequencies and
complete size/cost/memory comparison remain open. The 50/50 weight and lambda
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
