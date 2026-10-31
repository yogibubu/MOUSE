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

The original 44-state manifest and all dispositions are retained. The active scope was reduced after baseline convergence inspection to 14 states on ten molecules. Ten states pass stationarity and available-root audits; four remain under calculation. Accuracy labels remain sealed until prediction closure. This reduced/adapted set is not wholly blind. TD-DFTB/DeltaDFTB state assignments, full lambda/anchor sensitivity and complete derivative/memory costs remain unfinished.

Internal-coordinate results against CCSDR(3)/def2-TZVPP are available for three minima. Nitroxyl harmonic-frequency comparisons remain provisional by frequency order until mode-vector confirmation. These physical comparisons must be kept separate from numerical validation of the analytic derivatives.

## Revision of 31 October 2026

Code commit `5620e2d` includes spin-pure ROKS/DeltaSCF, analytic nuclear response, APOC/LINK integration, batched orbital response and assigned-NTO initialization. Main Sections 2.6--2.7 and 4.4 and SI Section 8 explain the solver, amended identity policy and measured timing scope. Seven scientific figures use archived results or labelled schematics. The obsolete static-response accuracy and timing comparisons have been withdrawn as TDA results.
