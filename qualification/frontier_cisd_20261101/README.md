# Frozen spin-adapted frontier CISD pilot

This research candidate is separate from the production relative-parent model.
It retains native GFN2 spatial ground orbitals, CAM-derived orbital-energy
references, lambda 0.9 on response exchange and fixed F/Q weights 0.5/0.5.
No excitation-energy offset is fitted or added.

`protocol.json` was frozen in commit 7666f7a before two prospective predictions.
The nine earlier states are exposed development. Acetone and glyoxal were
prospective for this frozen model, but historically exposed in older campaigns;
this is not a truly blind dataset. `score.py` reads labels only after prediction.

All eleven states converge and pass the fixed wavefunction assignment checks.
The total MAE is 0.216744 eV, maximum absolute error 0.551893 eV. The unchanged
0.30/0.50 eV accuracy gate is **not passed**. The prospective-only MAE is
0.384857 eV. The source/data retain glyoxal's full discrepancy. This candidate
is not promoted as a universal n-to-pi correction.

The native-orbital TDA control uses the exact same prepared frozen model and
range-separated exchange operator. All individually matched spatial MO overlaps
exceed 1-1e-7. Its eleven-state MAE is 1.162559 eV, but its two prospective
states alone give 0.420042 eV. This is one same-kernel control, not a claim of
superiority over all calibrated TDA approaches or Grimme's published methods.

The primary gap is a full-SI excited eigenvalue minus the independently
stationary ground-functional energy. Auxiliary SI ground lowering is reported
separately, not subtracted as a fitted offset. This state-specific composite is
not a balanced common ground/excited eigenvalue Hamiltonian. The full-SI-ground
referenced control is also retained in every result.

`nuclear.py` includes complete first/second ground and ROKS KKT response,
singular-safe Slater-Condon element jets, exact four-spin singlets and their
Heisenberg spin completion, metric and generalized eigenvector response. Its
anchored PES freezes auxiliary orbitals and physical off-diagonal operators in
the co-rotating orthonormal AO frame. It does not silently omit response from a
claimed re-canonicalized auxiliary model; see `nuclear_protocol.json`.

Complete Criegee and Furofuran Hessians pass independent reconverged molecular
energy validation: maximum extrapolated directional curvature discrepancies
1.89e-7 and 4.03e-7 hartree/bohr². The implementation itself uses no finite
nuclear steps. `parallel_hessian.py` preserves the completed sequential
Furofuran prefix and computes only missing columns on the verified remote
host. Its 126.83 s timing is parallel completion only, not a full comparable
single-worker benchmark. Raw result certificates retain worker counts and
sources. The derivative evidence validates this explicitly anchored PES,
not its excitation-energy accuracy.

`frontend.py` recovers both Criegee and both Furofuran ROKS states from four
canonical promotions with zero TDA-amplitude reads and zero new DFT/SCC calls.
No numerical match to reference excitation energies is used to find roots.

Portable NumPy/SciPy state-interaction primitives are available in
`matrix_gfn2.state_interaction`; the three public-module regressions pass on
MAC-STUDIO, including singular-overlap mixed derivatives. The portable algebra
is not by itself a complete public molecular-surface API. The complete
prepared molecular workflow remains this explicitly isolated research driver.

Large response tensors and reference inputs remain in the remote study roots;
`runtime_and_input_identity.json` records their hashes and exact paths. The
ported Python source changes only the explicit host assertion and workspace
prefix; the compiled native extension was rebuilt for the target architecture.
No heavy or electronic calculation ran on the laptop.
