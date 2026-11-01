# MOUSE semiempirical OO-DFT manuscript

The manuscript distinguishes the original screened-even OO-CID model from the common-reference repair motivated by ethylene. The latter has one explicitly calibrated bath-screening efficiency, **eta = 0.60**; F/Q weights remain 50/50, exchange-response lambda remains 0.9, and the energy offset is zero.

| Corrected exposed development set | States | MAE / eV | Maximum error / eV |
|---|---:|---:|---:|
| pi-to-pi* | 9 | 0.283 | 0.482 |
| n-to-pi* | 14 | 0.175 | 0.381 |
| All | 23 | 0.217 | 0.482 |

All 23 states in 21 molecules meet the individual 0.5 eV criterion. Ethylene is 8.206 eV against 7.897 eV, reducing its error from +1.182 to +0.309 eV. The freshly recomputed TDA control uses the same corrected native GFN2 orbitals and reference response: MAE 0.781 eV overall, 1.726 eV for pi-to-pi*, and 0.174 eV for n-to-pi*. Thus the mean accuracy gain is concentrated in pi-to-pi*.

Eta is calibrated using furan V010 and furofuran C002. The other 21 states give 0.192 eV MAE, but all 23 labels were exposed during development: this is **not blind validation or a universal-accuracy claim**. The original twenty-state result (0.225 eV MAE) and its prospective extension are retained as separate model generations. Earlier physical minima and frequencies are not attributed to the corrected model.

## Read

- `main.pdf` / `main.tex`: current 16-page method manuscript, with Levi's saddle-optimization motivation and positioning relative to Grimme's simplified response methods and other semiempirical alternatives.
- `supporting_information.pdf`: complete original and corrected 23-state tables and separately identified earlier-model evidence.
- `data/screened_cid/`: final predictions, aggregate scores, derivative certificates and Jmol coordinate/render checksums.
- `figures/screened_cid_validation.pdf`: all seven pi-to-pi* targets and thirteen controls, before/after the new construction.
- `figures/screened_cid_structures_jmol.pdf`: actual archived 3D Jmol renders of all twenty-one molecular anchors.
- `figures/screened_method_workflow.pdf`: current prepared-model workflow.
- `README_development_history.md`: the earlier package notes, retained as history.

## Derivatives and automatic integration

The corrected ethylene full analytic Hessian takes 10.15 s in one two-thread MAC-STUDIO call. Independent scalar-energy Richardson checks give maximum gradient error 1.73e-11 Ha/bohr and curvature error 6.56e-8 Ha/bohr². Its coherent singlet has S² = 0. All 23 corrected energies pass fresh-process spin and restart checks without DFT imports, TDA solves or new SCC calls. The corrected ethylene LINK check also passes cold start, accepted restart and displaced-trial rollback.

The following larger Hessian qualifications refer to the original screened-even predecessor:

Full analytic Hessians are qualified for Criegee's two states (15 coordinates), furan (27) and uracil (36). Independent reconverged energy differences at two step sizes, including an out-of-plane displacement, give maximum Richardson errors of 1.42e-10 Ha/bohr for gradients and 3.72e-7 Ha/bohr² for curvature. Finite differences are validation only. A direct public-API Hessian agrees with the assembled matrices within 2.16e-9 Ha/bohr².

For that predecessor, the public model loader and unchanged LINK `nosi_50_50` selector accept the new schema automatically. Fresh-process checks cover all twenty energies, coherent singlet spin purity and restart, with no DFT import, backend TDA input or new SCC driver call. The LINK pi state passes cold start, accepted-only restart and displaced-trial rollback.

The native energy, spin and nuclear-derivative primitives are reused. The final screening, auxiliary orbitals and CSF list are frozen in a co-rotating orthonormal AO frame. The analytic response differentiates this explicitly defined anchored surface, including first/second KKT, metric, projector and eigenvector response.

Exact auxiliary bitstring contraction, cached NumPy contraction programs, zero-term omission and Gram-block reuse are explained in the main text. Prepared energies take 1.7--9.0 s in the measured two-thread checks; the current dense direct Criegee Hessian takes 418 s and about 1 GiB. No vertical speed advantage over TDA is claimed.

## Rebuild current figures and PDFs

The figures use only the bundled data and archived Jmol PNGs; no electronic calculation or new Jmol optimization is launched.

```sh
python make_screened_cid_assets.py
python make_screened_workflow.py
python make_ethylene_repair_assets.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supporting_information.tex
```

Historical asset scripts rebuild earlier generations and should not overwrite the current model-generation data. Rendering and source checksums are recorded in `MANIFEST.json`.

## Provenance

The candidate choice is frozen in `2d59ab8`; the public response is `81897cf`; exact contraction/metric acceleration is `e572d50`; final molecular/runtime qualification is `358ae21`, with complete logs in `7c5c7f9`.

The associated repository archive `qualification/pi_pi_transfer_repair_20261101` retains 981 verified transferred artifacts, 21 matching public implementation files, three portable numerical examples and the full remote inventory. Electronic calculations ran only on verified Oracle or MAC-STUDIO. The laptop performed orchestration, file checks, editing, plotting and PDF review.
