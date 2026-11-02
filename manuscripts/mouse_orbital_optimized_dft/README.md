# MOUSE semiempirical OO-DFT manuscript

The current candidate adds zero-frequency static self-energy matching to the spectral-block OO-CID construction. The complete interacting auxiliary block is used, without an additional fitted coefficient. It uses one calibrated bath efficiency, **eta = 0.60**; F/Q weights remain 50/50, exchange-response lambda remains 0.9, and the energy offset is zero.

| Current exposed development set | States | MAE / eV | Maximum error / eV |
|---|---:|---:|---:|
| pi-to-pi* | 12 | 0.209 | 0.478 |
| n-to-pi* | 14 | 0.160 | 0.411 |
| All | 26 | 0.183 | 0.478 |

All 26 states in 24 molecules meet the individual 0.5 eV criterion. Matched TDA uses the same corrected native GFN2 orbitals and reference response: MAE 0.789 eV overall, 1.514 eV for pi-to-pi*, and 0.168 eV for n-to-pi*. The OO improvement is concentrated in pi-to-pi*; The static-matched candidate also slightly improves the mean n-to-pi* error relative to matched TDA.

The three added molecules are cyclopropene, cyclopentadiene and pyrrole. Their original fixed-model predictions were archived before numerical reference extraction. The pyrrole discrepancy motivated a generic spectral-block amendment, applied to every unresolved orbital of every molecule. Pyrrole now gives 6.302 eV versus 6.247 eV, with no retuning of eta. All 26 labels are exposed: this is **not blind validation or a universal-accuracy claim**. Eta was calibrated only on furan V010 and furofuran C002. Original 20- and 23-state generations and their predictions remain separately archived; earlier physical minima are not assigned to the latest candidate.

Beta-carotene and chlorophyllide a (Mg and conjugated substituents retained, phytyl chain removed) have archived structures, Jmol renders and prepared references. The coherent screening correction gives converged unsubtracted beta-carotene roots 1.95873, 2.34896 and 2.47471 eV at N=W=13. This qualification belongs to the unsubtracted Hamiltonian. Static matching preserves dark-below-bright ordering but changes the spectrum: its lowest root is 2.244 eV and its dominant absorbing root is root 4 at 3.031 eV (f=5.00); the static-matched beta-carotene frontier sequence is not yet qualified. Chlorophyllide's four ROKS targets are stationary. Its first two static-matched gaps meet the two-increment criterion over N=14,15,16, while the full higher spectrum remains under qualification. Twenty roots at N=16 recover minimum native-target coverage 0.527. Neither pigment enters the 26-state accuracy pool; pigment nuclear derivatives are not qualified.

## Read

- `main.pdf` / `main.tex`: current method manuscript, with Levi's saddle-optimization motivation and positioning relative to Grimme's simplified response methods and other semiempirical alternatives.
- `supporting_information.pdf`: complete original, corrected 23-state and current 26-state tables and separately identified earlier-model evidence.
- `data/screened_cid/`: final predictions, aggregate scores, derivative certificates and Jmol coordinate/render checksums.
- `figures/screened_cid_validation.pdf`: all seven pi-to-pi* targets and thirteen controls, before/after the new construction.
- `figures/screened_cid_structures_jmol.pdf`: actual archived 3D Jmol renders of all twenty-four small-molecule anchors.
- `figures/pigment_test_structures_jmol.pdf`: beta-carotene and chlorophyllide a, explicitly labelled as preparation.
- `figures/spectral_block_validation.pdf`: complete latest 26-state comparison.
- `data/spectral_blocks/`: current ledger, amendment, original additional predictions and pigment protocol.
- `figures/screened_method_workflow.pdf`: current prepared-model workflow.
- `README_development_history.md`: the earlier package notes, retained as history.

## Derivatives and automatic integration

The static-matched full analytic Hessians are qualified for ethylene and furan. Independent scalar-energy Richardson checks give maximum curvature errors 3.07e-8 and 6.50e-8 Ha/bohr², respectively. Ethylene takes 9.82 s with serial response columns and 6.10 s with four workers; furan takes 1386.40 s serially and 297.18 s with eight workers. Complete parallel and serial energy, gradient and Hessian arrays agree exactly. Times exclude reference preparation and do not establish an advantage over TDA. All 24 current models pass public-API energy, spin and restart checks without DFT imports, TDA solves or new SCC calls. LINK cold start, accepted restart and displaced-trial rollback also pass with static matching enabled.

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
python make_spectral_block_assets.py
python make_pigment_structure_plate.py
python make_carotene_ordering_assets.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supporting_information.tex
```

Historical asset scripts rebuild earlier generations and should not overwrite the current model-generation data. Rendering and source checksums are recorded in `MANIFEST.json`.

## Provenance

The candidate choice is frozen in `2d59ab8`; the public response is `81897cf`; exact contraction/metric acceleration is `e572d50`; final molecular/runtime qualification is `358ae21`, with complete logs in `7c5c7f9`.

The associated repository archive `qualification/pi_pi_transfer_repair_20261101` retains 981 verified transferred artifacts, 21 matching public implementation files, three portable numerical examples and the full remote inventory. Electronic calculations ran only on verified Oracle or MAC-STUDIO. The laptop performed orchestration, file checks, editing, plotting and PDF review.

The pigment implementation uses full-quadrature analytic orbital Hessian actions with an RI preconditioner, cached constraint cofactors, common-core/active-only integral transformations and exact fixed-bra contractions. `data/pigments/ordering_summary.json` and `figures/beta_carotene_state_ordering.pdf` retain the complete configuration-space sequence; executable qualification and compact matrices are in `qualification/closing_validation_20261101/results/pigments/`. The final affected regressions (20 tests, including target-subspace and independent phase checks), fixed-bra checks up to sixteen active electrons per spin and small-molecule analytic response controls pass on MAC-STUDIO. Reference preparation, screening, loading and runtime CI costs must be reported separately; no all-inclusive speedup over TDA is established.

Implementation, exact accelerations and archived carotene matrices: commit `3d04cc5` (on top of `d05afa6`).

`figures/beta_carotene_convergence.pdf` compares the coherent W=N sequence with the independent W=16 control; `tables/beta_carotene_expansion.tex` records the lowest three roots, including intensity exchange. Full matrices/checkpoints above 90 MiB remain on the verified compute host: `.remote.json` receipts record their paths, sizes and SHA-256 hashes, and compact lowest-twenty-state exports accompany them. Original large checkpoints are preserved. The manuscript data folder contains compact scalar views linked to the complete qualification files by SHA-256.

Larger-window evidence and fragmented-target tracking: `bb3435f`; residual trust steps and sparse spin recoupling: `e3c005a`. These supersede the earlier carotene-only ordering example while retaining its original results in the SI.

Final exact CI acceleration: `9248a39`; phase consistency and cached native-energy reuse: `8d44bb3`; coherent convergence protocol and audits: `cc1a5c2`; complete evidence: `0fb963f`. The original independent-W sensitivity criterion was superseded because W changes the active partition; its unsuccessful sensitivity checks remain archived in `beta_convergence_assessment_v1.json`. The final criterion is recorded in `beta_convergence_protocol_v2.json` and the passing verdict in `beta_convergence_assessment.json`.

Current static-response implementation: `7600833`; general 128-bit CSF contraction and cache: `67eb32f`; rank-revealing Davidson and target observables: `88a8314`; analytic Hessian column parallelism and LINK integration: `b1ad577`.

## Separate full-DFT response implementation

The PySCF adapter supplies fixed-occupation unrestricted DeltaSCF and common-orbital singlet ROKS, including complete analytic nuclear Hessians and moving Becke-grid response. It is retained as a companion implementation and is intentionally deferred from the present semiempirical manuscript. Sources, full arrays, checkpoints and checksums are in `qualification/dft_roks_response_20261102`; the general derivation is in `packages/matrix-pyscf/OO_RESPONSE.md`. The energy remains the approximate common-orbital ROKS multiplet sum, with no fitted offset.
