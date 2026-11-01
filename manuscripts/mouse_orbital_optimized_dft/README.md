# MOUSE semiempirical OO-DFT manuscript

The main manuscript now describes the final screened symmetry-augmented OO-CID candidate throughout. The former two-parent PBE0 model, earlier numerical cohorts, geometry/frequency comparisons and model-development history are retained separately in the Supporting Information and `development_manuscript_20261101.tex`.

The current assessment contains **20 exposed development states in 18 molecules**:

| Set | States | MAE / eV | Maximum error / eV |
|---|---:|---:|---:|
| pi-to-pi* | 7 | 0.180 | 0.349 |
| n-to-pi* | 13 | 0.250 | 0.498 |
| All | 20 | 0.225 | 0.498 |

All twenty states meet the unchanged individual 0.5 eV criterion. The same-native-orbital TDA control has 0.925 eV MAE. The reference weights remain 50/50, exchange-response lambda is 0.9, and there is no excitation-energy offset. Lambda does not scale orbital gaps.

These are model-development results, not a blind benchmark or universal-accuracy claim. The final model has not yet been assessed against high-level excited-state equilibrium geometries/frequencies or a broad independent CT/Rydberg set. Earlier physical results are not attributed to it.

## Read

- `main.pdf` / `main.tex`: current 12-page method manuscript, with Levi's saddle-optimization motivation and positioning relative to Grimme's simplified response methods and other semiempirical alternatives.
- `supporting_information.pdf`: complete current twenty-state table and separately identified earlier-model evidence.
- `data/screened_cid/`: final predictions, aggregate scores, derivative certificates and Jmol coordinate/render checksums.
- `figures/screened_cid_validation.pdf`: all seven pi-to-pi* targets and thirteen controls, before/after the new construction.
- `figures/screened_cid_structures_jmol.pdf`: actual archived 3D Jmol renders of all eighteen current molecular anchors.
- `figures/screened_method_workflow.pdf`: current prepared-model workflow.
- `README_development_history.md`: the earlier package notes, retained as history.

## Derivatives and automatic integration

Full analytic Hessians are qualified for Criegee's two states (15 coordinates), furan (27) and uracil (36). Independent reconverged energy differences at two step sizes, including an out-of-plane displacement, give maximum Richardson errors of 1.42e-10 Ha/bohr for gradients and 3.72e-7 Ha/bohr² for curvature. Finite differences are validation only. A direct public-API Hessian agrees with the assembled matrices within 2.16e-9 Ha/bohr².

The public model loader and unchanged LINK `nosi_50_50` selector accept the new schema automatically. Fresh-process checks cover all twenty energies, coherent singlet spin purity and restart, with no DFT import, backend TDA input or new SCC driver call. The LINK pi state passes cold start, accepted-only restart and displaced-trial rollback.

The native energy, spin and nuclear-derivative primitives are reused. The final screening, auxiliary orbitals and CSF list are frozen in a co-rotating orthonormal AO frame. The analytic response differentiates this explicitly defined anchored surface, including first/second KKT, metric, projector and eigenvector response.

Exact auxiliary bitstring contraction, cached NumPy contraction programs, zero-term omission and Gram-block reuse are explained in the main text. Prepared energies take 1.7--9.0 s in the measured two-thread checks; the current dense direct Criegee Hessian takes 418 s and about 1 GiB. No vertical speed advantage over TDA is claimed.

## Rebuild current figures and PDFs

The figures use only the bundled data and archived Jmol PNGs; no electronic calculation or new Jmol optimization is launched.

```sh
python make_screened_cid_assets.py
python make_screened_workflow.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supporting_information.tex
```

Historical asset scripts rebuild earlier generations and should not overwrite the current final-model data. Rendering and source checksums are recorded in `MANIFEST.json`.

## Provenance

The candidate choice is frozen in `2d59ab8`; the public response is `81897cf`; exact contraction/metric acceleration is `e572d50`; final molecular/runtime qualification is `358ae21`, with complete logs in `7c5c7f9`.

The associated repository archive `qualification/pi_pi_transfer_repair_20261101` retains 981 verified transferred artifacts, 21 matching public implementation files, three portable numerical examples and the full remote inventory. Electronic calculations ran only on verified Oracle or MAC-STUDIO. The laptop performed orchestration, file checks, editing, plotting and PDF review.
