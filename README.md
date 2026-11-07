# MOUSE

**Molecular Orbital Utility for Semiempirical Explorations**

MOUSE is a tight-binding platform for molecular geometry optimization,
ground- and excited-state derivatives, and subtractive QM/QM′ models.
It combines FAST-XTB and FAST-TDA with ORACLE molecular perception,
SMITH coordinates, the LINK optimizer and APOC electronic-state tracking.

This standalone distribution includes these components and their complete
mandatory internal Python dependencies. Installation needs no private
repository and no MATRIX checkout. The original `matrix_*` Python module
names are retained so that published calculation scripts remain usable.

The public standalone release contains runtime source, examples and compact
metadata only. Qualification campaigns, checkpoints, raw numerical matrices,
manuscript build trees and submission archives remain outside this repository
and are not required to install or run MOUSE.

## Install from source

Use Python 3.11–3.13 and a C++17 compiler. macOS uses the system Accelerate
framework. Linux needs development headers/libraries for BLAS and LAPACK
(or OpenBLAS). On Debian/Ubuntu, install `build-essential libblas-dev
liblapack-dev python3-dev python3-venv` before building.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python install.py
python examples/ground_state.py
python examples/excited_state.py
python examples/geometry.py
```

The installer builds and installs all bundled internal packages together,
then resolves the public NumPy/SciPy and other Python dependencies, including the PyTorch
runtime used by the analytic ALPB response. No
external electronic-structure executable is required for native GFN2-xTB
or TDA-xTB calculations. Adapters are included; their optional external
programs and licenses must be supplied separately when those adapters are used.

## Install a prebuilt release

Download the release archive matching your Python version and operating
system, extract it and activate a fresh virtual environment. Then run:

```bash
python install.py --offline
```

The platform archive includes both the internal wheels and external Python
dependency wheels. The supplied macOS ARM64/CPython 3.13 archive requires
macOS 14 or newer (the bundled ALPB numerical runtime requirement).
A macOS ARM64 archive cannot be used on Linux, Intel
macOS or another CPython version. The source distribution is portable to
the supported systems with the build prerequisites above.

## Capabilities and scope

- Resident GFN2-xTB ground-state energies, gradients and analytic Hessians.
- TDA-xTB and spin-flip response, including analytic derivative operators.
- Ground-reference ALPB solvent contributions in the same derivative framework.
- Sparse coordinate operations and bounded-memory optimization through LINK.
- State identity and transition-density overlap through APOC.
- Subtractive QM/QM′ energy, gradient and Hessian assembly with explicit caps.
- Immutable optional parameter profiles, including the one-scale,
  oxygen-centered water example described in the accompanying work.

The compact TDA example illustrates the derivative API. Its raw native
energy gaps are model dependent and are not an optical accuracy benchmark;
QM/QM′ replaces the low-level chromophore energy by a high-level term.

The water profile is an example of parameter development with analytic
derivatives. Standard GFN2 remains the default; extensive parametrization
and general water interaction energies are outside that calibration example.
The near-linear timings reported for long chains concern the coordinate
path. Dense molecular electronic response has a separate size dependence.

The complete optimization and vibrational applications in the manuscript
reach 149 atoms. The chlorophyll infrared example demonstrates feasibility.
The hydrated CT controls concern intramolecular chromophore excitations.

## Connected exploration and spectroscopy tools

SEEKER, TRINITY and SATI are separate packages and are not included in this
standalone distribution. The MOUSE examples cover electronic derivatives
and geometry optimization. Reproducing the connected vibrational and vibronic
workflows also requires TRINITY/SATI.

The accompanying manuscript cites the exploration method in
[Digital Discovery 2022](https://doi.org/10.1039/D2DD00070A) and the
[TRINITY framework](https://github.com/yogibubu/TRINITY-coupled-oscillators).

## Build a release locally

```bash
python -m pip install build setuptools==84.0.0 wheel==0.48.0 numpy==2.3.5
python tools/build_wheels.py --download-dependencies
```

Builds run locally and consume no GitHub Actions quota. Keep the generated
wheels in `dist/wheels` for installation; do not mix native wheels built
for different operating systems or Python versions in an offline archive.

## Reproducibility and citation

`SOURCE_RECORD.json` records the selected package source revision and a
SHA-256 digest for every original source file. `RELEASE_RECORD.json`
records the distributable files, platform and installation/example results.
`CITATION.cff` supplies software citation metadata. The manuscript and
Supporting Information describe the scientific models and numerical checks.

Repository: https://github.com/yogibubu/MOUSE

Releases: https://github.com/yogibubu/MOUSE/releases

## License

Original code is distributed under the BSD 3-Clause license in `LICENSE`.
Third-party software retains its own license. External dependency wheels
include the corresponding license metadata; they are not relicensed as MOUSE.

## Experimental spin-polarized DeltaSCF

The uracil/uridine qualification and timing comparison is in
[`qualification/uracil_uridine_20261029/REPORT.md`](qualification/uracil_uridine_20261029/REPORT.md).
The ground-state GFN2 energy/Fock and derivative kernels are reused with explicit
non-Aufbau occupations. ROKS adds the two-determinant singlet functional and its
shared-orbital response. This does not make an unchanged ground-state Hessian a
valid excited-state Hessian.

LINK's experimental `CappedQMComponent` and `capped_subtractive` optimizer path
accept `QMScanBackend(native_excited_method='roks',
deltascf_promotion=(hole, particle), electronic_state=1, name='matrix_gfn2')`.
Orbital indices are zero-based and must be assigned independently in the full
and capped systems. `native_excited_method='deltascf'` retains a spin-contaminated
determinant; only `roks` supplies the pure S=0 CSF. The native excited components
use zero temperature, analytic gradients/Hessians and APOC tracking against the
previous accepted optimizer state. Trial points do not advance it.

For a ROKS saddle search, pass `orbital_saddle_index=m`: this selects generalized
mode following using the analytic orbital Hessian. The index is a physical
state-search input, independently of TDA root numbers; success requires the
requested index and orbital stationarity. Without this field the experimental
least-squares stationarity solver remains available. API-level direct searches
use `evaluate_gfn2_roks_singlet(..., orbital_solver='gmf', saddle_index=m)` and
optionally `freeze_and_release=True`. GMF replaces excited-state SCF iteration;
it still reuses the ground-state energy/Fock and derivative kernels.

Direct standalone LINK scan/optimizer routes reject these flags rather than evaluating
the ground state. Molecular qualification does not support switching production
QM/QM' or spectra to ROKS by default; transition intensities still require the
existing TDA/TD-DFT path.

MOUSE now exposes `evaluate_gfn2_deltascf` and `run_deltascf`. The molecular
route adds the published on-site, shell-resolved spGFN spin-polarization energy
to resident GFN2, builds distinct alpha/beta Fock matrices, and relaxes both
orbital sets using fixed-reference maximum overlap (IMOM). Ground and excited
energies use the same spin-polarized functional. Target occupations conserve
total electron number and may change the nominal spin sector.

The shell constants are pinned and checksummed in
`packages/matrix-gfn2/src/matrix_gfn2/data/spin_constants.toml`, with source and
license provenance alongside the table. Actual s/p/d shell labels determine
parameter order; orbital index order is never assumed to be s,p,d.

```python
from matrix_gfn2 import evaluate_gfn2_deltascf
result = evaluate_gfn2_deltascf(
    calculator, xyz_angstrom, alpha_occupations, beta_occupations,
    ground_state=retained_ground_result,
)
print(result.excitation_energy_hartree, result.spin_squared)
descriptor = result.to_apoc(state_id=1)
```

Supply a zero-temperature, gas-phase GFN2 calculator and binary spin
occupations of length nmo. The present molecular adapter supports fewer than
64 atoms and rejects NN overlays, implicit solvent, point-charge embedding and
symmetry constraints. It provides analytic nuclear gradients (pass
`gradient=True`) and the fully analytic orbital-response Hessian through
`evaluate_gfn2_deltascf_analytic_hessian`. A separate checkpointed finite-
difference Hessian is available for independent validation. GMF,
freeze-and-release and optical transition properties are not implemented.
An Ms=0 broken-symmetry determinant is not itself a spin-pure singlet.

APOC's `follow_deltascf_state` checks continuity using occupied-determinant
cross overlaps with the actual AO cross metric. `result.to_apoc()` retains
physical nuclear charges and explicit frozen valence cores. No TD transition
density is manufactured.

Nonconvergence raises `DeltaSCFConvergenceError` with the final trial result in
`exception.result`. `save_checkpoint(path)` records orbitals, occupations, AO
metric and the original IMOM reference. For a restart of `run_deltascf`, read
these arrays with `numpy.load`, retain `reference_coefficients`, and supply the
saved `coefficients` as `initial_coefficients` with the same geometry/model
callback. No production parameters or existing TDA APIs are changed.

The algebraic tests are in `qualification/test_deltascf.py` and
`qualification/test_roks.py`. Molecular gradients, analytic Hessians, spin
adaptation and checkpoint reuse were validated on Oracle, together with an
independent ORCA 6.1.1 native-spGFN2 triplet energy/gradient comparison. See
`qualification/DELTASCF_VALIDATION_20261029.md` for inputs, tolerances and scope.
These consistency checks do not establish excitation-energy accuracy against
experiment or high-level correlated theory. Source-only tests require an
existing compatible native extension for the package's other public imports.


## Spin-pure singlet and analytic Hessian

For one singly excited pair (two open orbitals), MOUSE constructs a normalized
S=0 CSF from two spin-flipped determinants with common orthonormal orbitals.
`evaluate_gfn2_roks_singlet` reoptimizes the multiplet-sum functional
`E_S = 2 E_M - E_T` with this shared orbital set. `spin_adapt_gfn2_singlet`
starts from a retained BS result and uses corresponding occupied alpha/beta
subspaces to construct the restricted seed; occupied MO indices are not treated
as persistent orbital identities. This is a spin-adapted variational
reoptimization. The approximate GFN2 ROKS energy is not an exact spin-projection
energy of an unrestricted DFT determinant.

```python
from matrix_gfn2 import (
    spin_adapt_gfn2_singlet, evaluate_gfn2_deltascf_analytic_hessian,
)
singlet = spin_adapt_gfn2_singlet(calculator, result, gradient=True)
hessian = evaluate_gfn2_deltascf_analytic_hessian(
    calculator, xyz_angstrom, *singlet.occupations,
    mode="roks", reference_result=singlet,
)
print(singlet.spin_squared)  # S=0 CSF, not the mixed auxiliary determinant
print(hessian.hessian_hartree_per_bohr2)
```

The analytic Hessian uses the complete native fixed-density second nuclear
block, analytic shell-spin and AO-metric derivatives, and an indefinite coupled
orbital response. It forms the Schur complement `E_RR - E_Rk E_kk^-1 E_kR`.
ROKS uses one shared orbital response, including its single orthonormality
multiplier. There are no displaced SCF calculations or numerical derivative
steps in this analytic path. Singular orbital responses raise explicitly;
no positive-definite replacement or hidden regularization is used. This is a
dense small-molecule reference implementation.

`mode="deltascf"` selects the unrestricted determinant Hessian. A supplied
reference must match geometry/occupations and have orbital residual below
1e-8. Default SCF tolerances can be too loose for this boundary: converge the
reference explicitly with `orbital_tolerance=1e-10`,
`density_tolerance=1e-10`, `energy_tolerance=1e-12` before requesting its Hessian.
ROKS accepts its own `orbital_tolerance` control.

For an independent central-difference check, call
`evaluate_gfn2_deltascf_hessian` with `mode="roks"` or `"deltascf"`, a
`step_bohr`, and a `checkpoint_dir`. It records each converged displaced
analytic gradient with retained basis/state metadata. Matching checkpoints
are loaded without another SCF run. `raw_hessian_hartree_per_bohr2` preserves
pre-symmetrization errors. The numerical and analytic APIs have distinct names
and method metadata.

APOC's `SpinAdaptedSinglet` and `follow_spin_adapted_singlet` use coherent
cross overlaps of both determinants, including their cross terms. Its physical
spin-resolved one-particle density is unpolarized; mixed/triplet auxiliary
densities are retained only to evaluate the ROKS functional.

The experimental `match_orbital_spectrum(..., reference_energy_scale=0.9)`
maps DFT reference energies onto retained GFN2 orbitals. Both
`replace_tda_reference_gaps` and `correct_roks_excitation` consume this same
mapping, replacing each SE gap by the scaled DFT gap without scaling the SE
kernel or relaxation. Hybrid nuclear derivatives additionally require explicit
derivatives of the spectral correction. See the
[remote uracil comparison](qualification/hybrid_spectrum_20261029/REPORT.md)
for the selected lambda, configuration-space limitations and accuracy results.
The optional `with_occupied_extent_scaling(mapping, source_radii_bohr,
reference_radii_bohr)` attaches a smooth, experimental ratio of matched orbital
spreads to the occupied reference levels, using one common virtual energy as
origin. TDA and ROKS then use the same transformed spectrum. The
[extent-scaling qualification](qualification/hybrid_spectrum_20261029/EXTENT_REPORT.md)
records uracil development results and an independent thymine check. This
energy correction does not yet supply analytic nuclear derivatives of the radii.
An explicit experimental PBE0 profile with `reference_energy_scale=0.8706205766726199`
and `with_occupied_extent_scaling(..., power=1.5)` calibrates one common scalar on
uracil and reduces the thymine held-out MAE to 0.091 eV. See the
[calibration and cross-validation report](qualification/hybrid_spectrum_20261029/CALIBRATION_REPORT.md);
this profile is not a production default or a functional-independent calibration.
The subsequent [five-molecule generalization test](qualification/hybrid_spectrum_20261029/generalization/REPORT.md)
fails: six identifiable held-out states have a 1.354 eV MAE with the spectral
correction, versus 0.446 eV for uncorrected SE ROKS on the same states. The
uracil/thymine calibration must not be treated as a generally accurate model.
A subsequent [parameter-free frozen-energy test](qualification/hybrid_spectrum_20261029/frozen_relaxation/REPORT.md)
uses a full frozen PBE0 ROKS excitation plus the retained SE orbital relaxation.
It reduces the previous six-state cohort MAE from 0.446 to 0.212 eV, but still
reverses the uracil/thymine state ordering. This energy-only research prototype
has not replaced the default method or the existing analytic SE derivatives.
The subsequent [full DFT relaxation diagnostic](qualification/hybrid_spectrum_20261029/dft_relaxation/REPORT.md)
converges all four uracil/thymine PBE0/aug-cc-pVDZ singlets on Oracle with
common orbitals, analytic orbital gradients and response Hessian-vector products.
Full DFT ROKS restores the state ordering: SE underestimates n→π* relaxation by
0.440/0.472 eV, while its π→π* relaxation agrees with DFT within 0.008 eV.
The remaining π→π* error of −0.359/−0.420 eV is already present in PBE0 ROKS.
The diagnostic certifies stationarity and CSF identity, not the orbital Hessian
index, and does not supply nuclear derivatives of the new hybrid energy.

The [30 October molecular validation](qualification/roks_repair_20261030/REPORT.md)
adds explicitly frozen DFT charge/spin response models. `FrozenResponseReference`
contains the shell spin matrix, charge correction and reference populations;
`FrozenROKSAnchor` adds a fixed, state-specific energy offset. With
`kind="relaxed_vertical"`, the offset matches a fully relaxed DFT excitation at
the anchor geometry. `evaluate_anchored_roks` and
`evaluate_anchored_roks_analytic_hessian` optimize and differentiate the same
frozen potential. Its derivatives are SE model derivatives, not DFT derivatives
or derivatives of parameters recalibrated at each geometry. Matching a vertical
energy does not by itself improve the forces.

`save_frozen_roks_anchor` / `load_frozen_roks_anchor` persist the complete model
without pickle. LINK's experimental `QMScanBackend(native_excited_method="roks",
native_roks_anchor_path=..., deltascf_promotion=(hole, particle),
orbital_saddle_index=index, electronic_state=1)` checks the file checksum and
includes it in restart identity. APOC follows the coherent singlet CSF. The
analytic Hessian includes frozen-response charge, spin and AO-metric terms.

The response uses the uracil exact-exchange factor **lambda=0.9 once**, not an
additional 0.9 scaling of the orbital gaps. The selected frozen-energy transfer
profile fails the independent ketene/diazomethane/nitroaniline accuracy gate;
it is not a general correction for n→pi* and does not replace the default method.
Relaxed DFT energy anchors are a separate local research model with an explicit
DFT cost per state. The report compares actual TDA on GFN2 orbitals, records
failed state assignments, and separates solver timings from DFT preparation.

The [independent no-offset campaign](qualification/roks_independent_20261030/REPORT.md)
persists a response functional directly with `save_response_reference` /
`load_response_reference`. Pass it as `spin_reference=` to ROKS and its analytic
Hessian. In LINK use `native_roks_response_path` instead of an energy anchor;
the two settings are mutually exclusive and model checksums enter restart
identity. Ground and excited energies are evaluated with the same corrected
functional. The eight qualified states on seven new molecules have a 0.370 eV
MAE (native ROKS: 0.483 eV), but n→pi* errors remain as large as 0.972 eV.
This is an experimental candidate, not a universal n→pi* correction.

`FrozenTensorResponseReference` adds rotationally invariant on-site sp-vector
and pp-quadrupole charge/spin terms; `FrozenFockResponseReference` adds a linear
shell potential. Both are variational and support analytic nuclear derivatives,
but the tested DFT-derived versions worsen development energies and are not
recommended accuracy profiles. A charge or Fock correction requires ground-state
relaxation under the same functional, which the ROKS evaluator performs.
Complete AO-pair response is investigated separately as a vertical-only research
diagnostic. Its nuclear derivatives are explicitly unsupported.

A second blind test on acetaldehyde, acetone and formamide rejects the complete
AO-pair response as a general accuracy improvement: MAE 0.161 eV versus
0.044 eV for native ROKS and 0.090 eV for the shell correction. These vertical
results do not extend the scope of the analytic nuclear Hessian implementation.

The subsequent [solution audit](qualification/roks_independent_20261030/SOLUTION_AUDIT_20261030.md)
keeps the user-selected 50/50 mean of complete and spectral ROKS gaps fixed.
On the prospective acrolein/pyrimidine/nitrosomethane test its MAE is 0.171 eV
(native 0.619, complete 0.319, spectral 0.128); it fails the original gate of
beating both components. Fitted-weight diagnostics are archived and unused.
Using DFT levels and response from the same stationary DFT density worsens
the exposed six-state mean from 0.129 to 0.194 eV despite independent response
audits passing. A research SA-REKS(2,2) solver includes analytic orbital and
occupation response with a Schur complement. Its carbonyl improvement fails
to transfer to acrolein and uracil. All changed-state and unconverged roots
are retained; no universal n-to-pi-star correction is established. These new
nonlocal/REKS prototypes have no nuclear derivative implementation. Their
analytic electronic orbital Hessian is not a nuclear Hessian.

The [EDFT-inspired functional experiment](qualification/roks_independent_20261030/EDFT_FUNCTIONAL_REPORT_20261030.md)
adds an explicit transition-density Coulomb coupling and the published fixed
density-driven correlation factor, with analytic orbital response and no
energy offset. It is a GFN2 research adaptation, not the full GX24 functional.
Its exposed eleven-state n-to-pi-star MAE improves from 0.156 eV for the fixed
mean to 0.129 eV, but a prospective test on cyclopentadienone, maleimide and
pyranone rejects promotion: 0.155 eV versus 0.076 eV for the unchanged 50/50
control. All three held-out states qualify. The combined fourteen-state
difference is only 0.004 eV. The uracil pi-to-pi-star guard also fails.
The RSH variants and failed continuations are retained. Cross-functional
tracking checks the previously qualified excited CSF and the original DFT
transition separately; the original native canonical determinant overlap is
a diagnostic. These new prototypes do not implement nuclear derivatives or
change the production default.

### Candidato ROKS 50/50: derivate nucleari locali

La media fissa delle energie ROKS completo/spettrale è disponibile come API
esplicita `evaluate_composite_roks`, con gradienti e
`evaluate_composite_roks_analytic_hessian`. I due stati sono ottimizzati e
seguiti separatamente; si mediano le Hessiane già rilassate, senza offset.
`FrozenNonlocalResponseReference` estende localmente i modelli AO-pair congelati
con trasporto molecolare co-rotante e derivate analitiche fino al secondo ordine.
Supporta basi s/p e geometrie non lineari, incluse quelle planari.

Validazione su Oracle: HNO e acetaldeide, tutte le colonne delle Hessiane a due
passi, errori del composito inferiori a 1,7×10⁻⁸ Ha/bohr²; 39 test superati.
È una superficie locale sperimentale, non una correzione universale n→π*.
Vedere [protocollo, API e limiti](qualification/roks_50_50_20261030/REPORT.md).

LINK supporta ora `native_excited_method="roks_50_50"` nelle scansioni dirette,
nel servizio di ottimizzazione residente e nei componenti QM/QM. Si indicano
`native_roks_composite_response_paths=(raw_path, spectral_path)` e
`native_roks_composite_saddle_indices=(raw_index, spectral_index)`, insieme alla
promozione orbitale. Il tracciamento e la ripresa sono separati per le due CSF;
soltanto i punti accettati avanzano i riferimenti. Energia, gradiente e Hessiana
analitica vengono mediati automaticamente con pesi fissi 1/2.
Validazione Oracle: 44 test, scansioni HNO/acetaldeide e ripresa dell'ottimizzatore.
Vedere [configurazione e controlli LINK](qualification/roks_link_50_50_20261030/REPORT.md).

La successiva prova di ottimizzazione 50/50 su HNO, acetaldeide e maleimmide
non certifica ancora minimi eccitati: arresto elettronico per HNO, due/tre modi
immaginari nei punti principali di acetaldeide/maleimmide, continuazioni non
convergenti. Il candidato resta verticale; vedere
[prova molecolare e limiti](qualification/roks_50_50_optimization_20261030/REPORT.md).
Il dispatch della Hessiana finale ora usa la superficie eccitata corretta;
47 test includono le regressioni del fallback TDA e della ripresa delle ancore.

Per continuazioni ROKS 50/50 si possono ora selezionare esplicitamente
`native_roks_composite_following="character"` oppure
`"character_adaptive_index"`. Il primo conserva il controllo APOC completo fra
passi accettati e usa, rispetto all'ancoraggio originale, la coppia di orbitali
aperti in una base AO ortogonalizzata e corotante. L'overlap completo con
l'ancoraggio resta registrato. Il secondo permette inoltre un solo tentativo
con indice orbitale adiacente, entro ±1 dall'indice iniziale, con gli stessi
controlli sullo stato. Il comportamento precedente resta il valore predefinito.
`native_roks_composite_ground_recovery="accepted_retry"` abilita un tentativo
aggiuntivo dello stato fondamentale dal checkpoint accettato se il riferimento
indipendente fallisce; le soglie residue rimangono identiche. Queste opzioni
cambiano il contratto di restart e richiedono una migrazione esplicita.
La [validazione delle continuazioni](qualification/roks_following_20261030/REPORT.md)
riporta separatamente i miglioramenti numerici e i limiti della superficie.

La [correzione numerica HNO](qualification/hno_fock_20261030/REPORT.md) evita di
riattivare l'annealing termico durante la ricertificazione Newton, corregge
l'eigensistema di uscita di un SCC diretto non convergente ed elimina il calcolo
S0 ausiliario alla conclusione ROKS. 64 test remoti e i controlli Hessiana-vettore
nei precedenti punti respinti passano; il rilassamento resta limitato dal
riferimento S0 quasi degenere e non produce un minimo qualificato.

Il riferimento `native_roks_composite_reference_policy="geometry_only"` elimina
la dipendenza da una soluzione SCC S0 durante la valutazione ROKS. Il problema
agli autovalori del core definisce soltanto la base di lavoro; energia assoluta,
Fock, gradiente e Hessiana provengono dal funzionale fisico completo. I profili
`local_csf` e `local_csf_adaptive_index` seguono le CSF fra punti accettati e
registrano separatamente l'evoluzione rispetto all'ancora iniziale.
`native_roks_composite_component_tracking="active_pair"` confronta la coppia
aperta fra le due Hamiltoniane, senza includere le rotazioni spettatrici chiuse.

L'opzione sperimentale `native_roks_composite_ground_surface="native_plus_gaps"`
usa invece la superficie
`E0_GFN2(R) + 0.5 * sum_i(E_ROKS_i(R) - E0_i(R))`.
Le tre soluzioni fondamentali ristrette sono ottimizzate direttamente, senza
SCC, e sono incluse nel restart accettato. Gradiente e Hessiana comprendono
anche tutte le derivate della sostituzione di S0: non è un offset costante.
L'opzione richiede `geometry_only`; cambia il contratto del checkpoint e
rifiuta gap di eccitazione non positivi in una delle due componenti.
Le API corrispondenti sono `reference_roks_to_native_ground` e
`ground_referenced_roks_analytic_hessian`.

La qualifica remota attuale comprende 70 test, incluse le derivate della nuova
superficie e il restart delle cinque soluzioni. HNO raggiunge un minimo con
Hessiana analitica positiva e gap positivi; questo risultato non risolve da solo
il problema universale n→π*. I protocolli e gli script delle verifiche sono in
`qualification/roks_direct_20261030/`.

Il candidato aggiornato usa
`native_roks_composite_orbital_coupling="relative_active_pair"`: conserva i due
overlap al quadrato h/h e p/p dell'ancora assegnata, lasciando rilassare le
orientazioni assolute e gli spettatori. In questo modo mantiene esattamente le
energie verticali del 50/50 indipendente. I vincoli definiscono una superficie
composita coerente nello stato; non aggiungono un termine energetico e non
usano le TBEs. La risposta orbitale/nucleare risolve il sistema KKT completo,
con eliminazione analitica degli orbitali già identici fra i due modelli.

La [qualifica del candidato a vincoli relativi](qualification/roks_relative_pair_20261030/REPORT.md)
comprende **86 test remoti**, cinque minimi con frequenze reali e verifica
indipendente di tutte le colonne della Hessiana per HNO e formammide.
Su 19 stati / 18 molecole il MAE è **0,1825 eV** (18 n→π*: 0,1757 eV).
Sulle quattro molecole prospettiche aggiunte il MAE è 0,3036 eV.
I vecchi confronti etichettati TDA sono ritirati: mancava un termine di
scambio nonlocale nella risposta statica. I controlli TDA corretti vengono
ricalcolati sugli stessi orbitali GFN2 corretti.
L'errore massimo residuo è 0,5349 eV: il candidato non è una correzione
universale dimostrata. Il confronto temporale completo con TDA resta da chiudere; sono
verificate la formulazione stazionaria e la disponibilità di
gradienti/Hessiane analitiche riutilizzando le
primitive del programma fondamentale.

Il primo avvio può ora leggere direttamente
`native_roks_composite_seed_paths=(raw_seed, spectral_seed)`, due checkpoint
ROKS già qualificati e coerenti con i rispettivi file di risposta.
LINK verifica checksum, modello, promozione e indici, inizializza APOC e dà
precedenza a una generazione accettata già esistente. Il restart conserva
le due CSF, i vincoli, i moltiplicatori e i tre stati S0; non occorre assegnare
manualmente `component.reference` o `component.anchor`.

L'[esempio eseguibile](examples/roks_relative_pair.py) valuta energia,
gradiente e, con `--hessian`, la Hessiana analitica. È protetto per l'esecuzione
sul solo host Oracle verificato e riprende automaticamente la geometria
accettata nelle invocazioni successive:

```bash
python examples/roks_relative_pair.py \
  --raw-response raw_response.npz --spectral-response spectral_response.npz \
  --raw-seed raw_vertical.npz --spectral-seed spectral_vertical.npz \
  --run-dir roks_run --hessian
```

Il backend richiede inoltre `native_roks_composite_following="character_newton_index"`,
`native_roks_composite_reference_policy="geometry_only"` e
`native_roks_composite_ground_surface="native_plus_gaps"`.
Pesi 50/50 e λ=0,9 restano invariati. La scelta è esplicita e il default
non cambia. Tutti i lavori elettronici, i benchmark e le ottimizzazioni
sono stati eseguiti su Oracle; sul MacBook sono stati effettuati soltanto
orchestrazione, controllo leggero dei sorgenti e trasferimento dei risultati.

### Solutore orbitale accelerato e selezione assegnata

La risposta orbitale analitica ora elabora insieme le perturbazioni di densità
e riusa la risposta di carica comune ai due spin. Sul medesimo checkpoint
Oracle (44 AO, 510 rotazioni), il Jacobiano passa da 4,138 a 0,242 s,
con differenza massima 8,88e-16; questo è un guadagno di kernel, non del
workflow completo. Le transizioni assegnate inizializzano direttamente i
modelli mediante NTO, con indice certificato e checkpoint conservati.
[Protocollo e limiti](qualification/state_following_20261031/PERFORMANCE.md).
