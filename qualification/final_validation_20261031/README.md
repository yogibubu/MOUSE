# Four-gate final validation

The frozen candidate uses equal F/Q weights, lambda 0.9, zero excited-state offset,
and the previously declared effective-transition-rank rule. Rank one retains the
original relative-pair single-CSF energy; ranks two through four use the public
full-active-frame multi-CSF surface. No validation score may select a new parameter.

Electronic work runs only on verified `oracle.sns.it`, with two threads and the
shared publication compute lock. The laptop performs orchestration, source and
hash checks, document preparation, and small reductions of transferred data.

## Executable work

1. `anchor_energy.py` reuses all six certified displaced DFT ground/response
   checkpoints, recalibrates complete F/Q operators and projected ERIs, certifies
   the three restricted S0 references, and compares excitation energies at the
   original common geometry. Original relative active frames are transported
   isometrically. This separates anchor dependence from moving the target nuclei.
2. `freeze_independent.py` fixes six states in five new molecules before their
   predictions. A repeated exposure audit excludes actual previous campaign
   paths, while omitting dependency-distribution test directories. Numerical
   reference labels are segregated. `independent.py` attempts every state using
   fixed symmetry/ordinal and projection gates. `independent_controls.py` obtains
   TDA controls on each corrected GFN2 ground's own orbitals. `score_independent.py`
   refuses to read labels before all six attempts close and their hashes lock.
   The cohort has four pi-pi states, one n-pi and one Rydberg state; CT remains a
   declared shortfall. Strong multiconfigurational and representation limitations
   must remain visible in the six-state coverage denominator.
3. `geometry.py` optimizes retained frozen surfaces with coherent state tracking,
   saves accepted geometry generations, and computes an analytic Hessian at a
   stationary geometry. Imaginary frequencies are retained. `compare_furan.py`
   compares five bond coordinates, five angles and fifteen published in-plane
   harmonic values with the EOM-CCSD reference, DOI 10.1063/1.1578051. Symmetry
   characters are verified, then frequencies are ordered within each irrep;
   displacement-overlap assignments are unavailable from that published table.
   The symmetric furan A1 structure is a nuclear saddle. This comparison must not
   be described as full-dimensional minimum/frequency validation.
4. `cost.py` measures energy, gradient, analytic Hessian and both-parent all-root
   TDA construction in separate fresh processes, three repetitions per case.
   Six existing molecular sizes are fixed, with rank and AO count reported.
   Model/ground loading and preparation costs remain separate. Previously missing
   preparation timings are unknown, never zero. Prepared TDA and one-state OO
   timings are not interchangeable end-to-end workflows; source preparation and
   full-process memory must be audited separately before a full cost claim.

`controller.py` executes the finite task list serially with checkpoints and
bounded attempts. Revision 2 schedules the independent science before the largest
repeated Hessian costs; the already-running geometry worker was allowed to finish.
No electronic checkpoint was restarted for this scheduling change.
`postprocess.py` waits for this finite prediction pass, computes matched TDA
controls serially under the same lock, then scores the locked ledger.
`collect.py` is read-only and never opens sealed labels. `archive.py` transfers
checksummed evidence, retains large preparation tensors remotely, and excludes
sealed numerical labels.

## Numerical setup and provenance

The first anchor setup attempts exposed missing directory/alias setup and the
need to canonicalize a directly minimized ground's intrasector Fock. All logs
are retained, and the completed calibration and ground checkpoints are reused.
Ammonia's C3v A1 state correlates uniquely with A-prime in PySCF's Cs working
subgroup; the explicit correlation was recorded before its response calculation.
This fixes a group-label API mismatch without changing the physical target.

`launch_manifest*.json`, per-prediction/property execution hashes,
`runtime_sources.json`, source histories and transfer manifests identify the
executed sources. `remote_snapshot` is explicitly an in-progress snapshot;
its existence does not imply the campaign is complete. Large arrays and backend
SCF checkpoints remain on Oracle with hashes. No numerical reference labels are
included in that snapshot.

## Subsequent numerical and sampling qualifications

- The original anchor is reconstructed algebraically with exactly identical
  F/Q operators for all three molecules. The native calibration densities agree
  with the direct restricted minima within 1e-12. This distinguishes anchor
  sensitivity from inadvertently changing the reference prescription.
- The most negative and highest positive vibrational directions at BOTH new
  stationary geometries pass independent displaced-gradient checks: maximum
  Hessian-vector errors 9.48e-9 (furan) and 1.44e-7 (cyclopentadiene) Ha/bohr².
- All representable primary states selected rank one. A separate two-state
  furofuran molecular challenge was frozen before its calculations; its two
  references also select rank one. Three previously unpredicted higher pi-pi
  states of benzoxadiazole were then frozen as a separate STATE-level challenge,
  reusing only state-independent ground and F/Q preparation. They are not
  presented as three new molecules or merged into the primary holdout silently.
  The numerical reference labels remain segregated for each cohort until its
  complete prediction ledger closes. Selection was independent of scored errors.
- `recover_anchor.py` checkpoints longer orbital continuation and uses a spectral
  trust-region step for the linearized orbital residual. It preserves the full
  active Gram constraint and original energy operators, state-fidelity guards,
  curvature certificate and zero offset. The saved orbital checkpoint was checked
  before changing the numerical solver; completed calibrations are retained.
- The first new TDA driver omitted the already-known nonlocal correction to the
  static response. `tda_operator.py` restores
  `A = A_static + 0.5*0.225*((ib|aj)-(ij|ab))`, matching the independently validated
  reduced-cohort operator. Static matrices are retained and reused, not scored
  as true TDA. Superseded response timings are kept separately. The corrected
  scorer checks the operator contract before accepting comparison values.
- Single-CSF Hessian timing now explicitly includes all three ground gradients.
  The earlier narrower measurements are archived separately and remeasured only
  for the corrected property scope. `cost_budget_amendment.json`, frozen before
  the largest-system Hessian measurement, caps that measurement at 600 seconds;
  a censored or >300-second first measurement is not redundantly repeated.
  This changes the cost-experiment budget, not the electronic candidate.

The new molecular geometries are rendered by actual Jmol in the manuscript's
independent-cohort plate. Combined with the existing plate, the two-page structure
artifact contains 34 molecules. Geometry hashes and renderer provenance are saved.
