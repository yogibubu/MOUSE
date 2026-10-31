# GFN2 TDA versus spin-pure orbital optimization: qualification, 29 October 2026

**The primary comparison uses the identical zero-temperature GFN2 ground-state orbitals, atoms and geometry in both methods. No DFT-orbital TDA result is used to establish an advantage. Accuracy is assessed before timings.**

The direct orbital method is ROKS-GFN2, with a two-determinant S=0 CSF and the approximate multiplet energy `2 E_mixed - E_triplet` at shared orbitals. It is not an exact projected energy of an arbitrary unrestricted determinant. The determinant-only DeltaSCF path is spin polarized but remains spin contaminated.

## Accuracy at the exact QUEST uracil geometry

The unmodified GFN2 parameters and shell spin constants are used; no excitation-specific fit or empirical correction was introduced. Both GFN2 calculations use the same retained canonical GFN2 reference and complete occupied/virtual space. The correlated TBE is the independent external reference.

| Energy / eV | nπ* | ππ* |
|---|---:|---:|
| Correlated QUEST TBE | 4.9350 | 5.3810 |
| Functional-consistent GFN2 singlet TDA **diagnostic** | 3.7278 | 4.8279 |
| ROKS-GFN2, explicit GMF saddle search | 4.3194 | 4.2762 |
| ROKS minus TBE | -0.6156 | -1.1048 |
| Functional TDA minus TBE | -1.2072 | -0.5531 |

ROKS improves nπ* and worsens ππ*. It reverses their order. The two-state mean absolute errors are approximately 0.860 eV for ROKS and 0.880 eV for functional TDA; a difference of 0.020 eV on two states does not establish a general advantage.

The functional TDA is a **new qualification diagnostic**, not the existing registered FAST-TDA model. Its singlet matrix is `A = orbital_gap + 2 K_charge`, where `K_charge` is the existing complete analytic GFN2 Fock-density Jacobian, evaluated on symmetric half transition densities. The shell spin kernel cancels between the alpha-alpha and alpha-beta singlet channels. Independent finite Fock-density derivatives agree within 6.7e-12 hartree; an independent finite orbital-energy curvature agrees with `4(A+B)` within 9.7e-8 hartree. This checks the construction, not broad molecular accuracy or transition properties.

The existing empirical sTDA construction is retained as a separate control. Applied directly to these GFN2 orbitals, its complete-space compact/constant-0.5-eV variant gives **-1.3763 and +0.2187 eV**. Those are not acceptable absolute excitation energies. The empirical screened J/K model is distinct from the parent GFN2 response; the literal std2 kernel also requires its own xtb4stda orbital reference for parity. One must not conclude that all TDA-GFN2 formulations are inaccurate from this incompatible absolute-energy control, or use it alone to claim a ROKS advantage. Its established use in low-level subtractive shifts is assessed separately below.

QUEST supplies vertical correlated electronic energies, not experimental band maxima. The benchmark uses its exact geometry, rather than comparing values at different optimized structures. The local metadata identify the composite CCSDT/small-basis plus CC3 basis-correction reference, repository commit and CC BY-SA license. Sources and machine-readable values: [accuracy_gfn2_only.json](accuracy_gfn2_only.json), [parent_response_diagnostic.json](parent_response_diagnostic.json), [parent_response_validation.json](parent_response_validation.json), [QUEST_Uracil.json](inputs/QUEST_Uracil.json), [license](inputs/QUEST_LICENSE). The archived DFT-orbital TDA CSV is background provenance only and is excluded from this primary comparison.

## Direct saddle search instead of excited-state SCF iteration

The new GMF solver reflects the orbital gradient along the lowest `m` eigenvectors of the analytic electronic Hessian and follows a fixed requested index. Steps update orthonormal orbitals by a matrix exponential. It does not minimize the electronic energy or the squared gradient. Success requires both a small orbital gradient and the requested number of negative eigenvalues. An optional freeze-and-release stage first relaxes the other orbitals while holding the two open orbitals fixed.

The saddle index belongs to orbital space, not nuclear coordinates. It is not a TDA root number. Approximate restricted ansatzes can also have excited local minima; the index alone does not identify a physical state. APOC overlaps and orbital/fragment character remain necessary.

At the original uracil S0 geometry the stationary nπ* and ππ* solutions have indices 1 and 2. GMF independently recovers nπ* at 4.280881 eV, residual below 1e-15, in about 3.2 s. The QUEST controls converge to indices 1 and 2 with residuals below 2e-10.

For uridine, ordinary IMOM and least-squares stationarity searches frequently fail or reach the wrong state. Freeze-and-release plus GMF finds nπ* at both required geometries:

| Fixed geometry | nπ* gap / eV | Orbital index | Maximum orbital residual / Eh |
|---|---:|---:|---:|
| S0 | 4.246264 | 1 | 1.03e-10 |
| Existing nπ* minimum | 3.338984 | 1 | 5.14e-11 |

Starting GMF from the failed least-squares trial does not recover this state; restarting from the canonical excitation and using freeze-and-release does. The failed trial and every GMF accepted iteration retain checkpoints. This is evidence for the importance of the initial constrained relaxation.

The uridine ππ* problem remains **unresolved**. One GMF run reaches an index-2 stationary solution but changes character and has CSF overlap about 0.011 with its seed; APOC correctly rejects it. An earlier stationary index-0 solution and its derivatives remain useful implementation controls, but are not accepted as a confirmed matched ππ* state. The earlier exploratory absorption shift -0.507 eV must therefore not be used as a qualified accuracy result. A four-step continuation toward the ππ* geometry is also rejected at its first step. We do not replace a failed state assignment with a numerical root label.

Artifacts: [gmf_summary.json](gmf_summary.json), [orbital_indices.json](orbital_indices.json), [gmf_character.json](gmf_character.json), per-state `results/*/gmf_*/history.json` and checkpoints.

## Uridine–uracil shifts and subtractive QM/QM′ replay

The original three geometries are retained: isolated uracil has its existing S0/excited structures; full uridine and its H-capped base use the same existing QM/QM′ structures as the previous B3LYP reference. These are vertical gaps, not new optimized geometries, 0–0 energies or vibronic spectra. All coordinate/source hashes are in `inputs/`.

The consistently identified nπ* comparisons are:

| Observable: uridine minus uracil / eV | ROKS-GFN2/GMF | Existing empirical TDA-GFN2 control | Full B3LYP reference shift |
|---|---:|---:|---:|
| Absorption at S0 | -0.0346 | +0.0811 | +0.1350 |
| Vertical emission at existing nπ* geometries | +0.0288 | +0.1002 | +0.1381 |

ROKS does not improve these environmental shifts: absorption has the wrong sign and the emission shift is underestimated. B3LYP here is a fixed-geometry comparison model, not an independent correlated uridine accuracy benchmark.

Using the unchanged capped-base high-level B3LYP gaps and replacing only the low-level difference gives an executable energy replay of `gap_high(model) + gap_low(real) - gap_low(model)`:

| nπ* observable / eV | ROKS low(real)-low(model) | Replayed QM/QM′ gap | Full B3LYP gap | Replay minus full |
|---|---:|---:|---:|---:|
| S0 absorption | +0.00227 | 4.66087 | 4.82090 | -0.16003 |
| Existing nπ* geometry | +0.05780 | 3.37530 | 3.46410 | -0.08880 |

This does not support switching the production low-level model to ROKS. The ππ* replay remains unqualified until state identity is resolved. No new B3LYP calculations were launched. Artifacts: [shifts_gmf.json](shifts_gmf.json), [comparison.json](comparison.json), [B3LYP reference provenance](inputs/b3lyp_reference.json).

## Analytic gradients, Hessians and timings

The nuclear Hessian is analytic: fixed-density native GFN2 second derivatives, spin/metric derivatives, and the orbital Schur complement `H_RR - H_Rk H_kk^-1 H_kR`. Indefinite orbital curvature is allowed. Singular response is exposed rather than silently regularized into a ground-state response. The GMF orbital Hessian is analytic too.

Vectorizing the orbital-pair contractions reduces the uracil analytic nuclear Hessian from about 24 s to **2.6–2.7 s**, with changes below 1.2e-15 Eh/bohr² relative to the completed earlier calculations. H2/formaldehyde saved-reference regression remains within 1.7e-11. On the 29-atom uridine implementation control the ROKS Hessian takes about **43 s**, with response residual and raw asymmetry below 1.1e-15. The properly found nπ* saddle at S0 takes about **45 s**, response residual below 9e-16.

Independent displaced molecular gradient points validate the analytic results. For the four uracil observables, energy-direction errors are below 6.3e-9 Eh/bohr and Hessian-action errors below 2.3e-8 Eh/bohr². The uridine nπ* S0 saddle has energy-direction error below 8.3e-10 and Hessian-action error below 1.1e-10. At the existing nπ* geometry the corresponding errors are below 1.8e-10 and 1.6e-10, with an analytic Hessian time of 39.9 s. Every displaced gradient is checkpointed; these finite differences validate the analytic implementation and are not how its Hessian is produced.

The existing **empirical** full-space TDA implementation is faster for vertical energies. On uracil it takes about 0.17 s for ten roots, versus about 3 s per initial ROKS state. On uridine, existing TDA takes about 10 s for ten roots. For the physically selected TDA root 3 its analytic derivative construction takes about 79 s and its full Hessian another 221 s. ROKS can thus save derivative time in this implementation, but these costs concern different response models, not a demonstration of equal-accuracy performance. The new functional-consistent TDA diagnostic currently has energies only; no Hessian speed comparison against that diagnostic is claimed.

Timings are wall times on shared `oracle.sns.it` with thread limits of two, not isolated hardware benchmarks or asymptotic scaling evidence. The implementation is a dense small-molecule path. Remote source snapshots reuse the installed Linux native extension; no heavy calculations ran on the laptop.

## Reusing the ground-state program and LINK integration

**Yes: the ground-state energy/Fock and derivative engine is reused directly.** Occupation control and an orbital saddle optimizer replace excited-state SCF iteration. ROKS additionally evaluates the mixed and triplet spin components at shared orbitals. One must not call an unchanged ground-state Hessian routine and interpret its result as the excited-state Hessian: the occupations, constraints, multiplet functional and indefinite orbital response must match the target state.

The experimental bundled LINK `CappedQMComponent`/`capped_subtractive` path supports:

```python
QMScanBackend(
    name='matrix_gfn2', electronic_state=1,
    native_excited_method='roks',
    deltascf_promotion=(hole, particle),  # independently assigned zero-based MOs
    orbital_saddle_index=m,            # explicit electronic Hessian index
)
```

It uses zero-temperature excited components, analytic gradients/Hessians, and projected orbitals/APOC following against the previous accepted geometry. Rejected trials do not advance the reference. Direct single-backend LINK scan/optimizer routes explicitly reject the new flags; they are not silently routed to a ground-state or TDA calculation. The general GFN2 API exposes `orbital_solver='gmf', saddle_index=m, freeze_and_release=True`.

Real remote integration controls verify three-component subtractive E/g/H cancellation, all three accepted-state histories, the actual LINK factory's temperature setting, and an uracil GMF component with index 1 and an analytic Hessian. They are implementation tests, not full uridine geometry optimizations or new cap-boundary accuracy validation. Existing TDA/TD-DFT remains available for transition intensities; no production default or spectral pipeline was switched.

22 algebraic/configuration tests pass. Files are uncommitted; no deployed runtime, release, commit or push was changed. Source hashes and remote/input provenance: [final_provenance.json](final_provenance.json). The remote campaign remains at `/home/enzo/MOUSE-deltascf-validation-20261029/uracil_uridine_20261029`.

Method sources: [Schmerwitz, Levi and Jónsson, GMF](https://arxiv.org/abs/2302.05912), [freeze-and-release](https://arxiv.org/abs/2501.18568), [ORCA DeltaSCF manual](https://www.faccts.de/docs/orca/6.1/manual/contents/spectroscopyproperties/deltascf.html), [QUESTDB](https://github.com/pfloos/QUESTDB), pinned commit `460482bcc6231dab41fe32188dc43959a7c2bdaa` with license and metadata retained in `inputs/`.
