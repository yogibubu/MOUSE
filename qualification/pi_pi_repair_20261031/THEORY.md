# Pi-to-pi-star repair: minimal spin-pure state interaction

This is a development experiment on already exposed reference energies, not a
new blind validation. The original 14-state predictions and their prediction
lock are preserved. Additional molecules are suspended.

## Diagnosis

The fixed projected DFT transitions have leading-NTO squared weights about
0.544 (cyclopentadiene), 0.533 (furan), 0.754 (phthalimide), 0.901 (adenine),
and 0.960 (syn-bimane). A one-pair singlet CSF cannot contain the missing
orthogonal transition pairs. The large errors coincide with strong missing
configuration weight; this is evidence of a limitation, not a proof that it
is the only error source. Simply inserting TDA off-diagonal response elements
between ROKS diagonal energies overcorrects several fixed-orbital energies.
Those matrices are retained only as diagnostics.

## Explicit candidate energy

Use one orthonormal spatial orbital set C and the singlet excitation CSFs
Phi_i(C) for distinct occupied-hole / virtual-particle pairs. Their real
linear combination is normalized and has exactly S=0. Define

    H_ii(C) = 2 E_Mi(C) - E_Ti(C)
    H_ij(C) = 2 (h_i p_i | h_j p_j) - (h_i h_j | p_i p_j), i != j
    E(C)    = selected eigenvalue of H(C).

The diagonal uses the unchanged spin-polarized, DFT-response-informed GFN2
ROKS parent. The off-diagonal is a wavefunction Slater-Condon coupling from the
already projected bare two-electron integrals. It is NOT the TDDFT kernel,
and NOT a claim to reproduce the old single-CSF functional. This is an
explicitly different approximate multistate functional. The use of
wavefunction couplings with DFT diagonal energies follows the minimal-space
state-interaction motivation in
https://doi.org/10.1038/s41524-021-00624-3 ; its correlation and double-counting
approximations require validation rather than an assumption of exactness.

The diagonal exchange-response lambda remains 0.9 (PBE0 exchange 0.225).
The new wavefunction coupling is unscaled bare Slater-Condon, as declared in
the protocol; it must not be described as the same exchange operator as TDA.
No mixing scale is fitted to energy errors. No additive excitation correction
is used. The two corrected parents still enter the vertical energy with
weights 0.5 and 0.5, with each parent's own certified S0 subtracted.

The CSF count is the smallest rank retaining at least 95% of the original
projected reference transition norm, capped at four in this experiment.
Selection uses the transition singular values, not target excitation energies
or an n/pi label. Consequently a general test of this candidate also has to
recompute B011 and B039; it cannot quietly retain their old single-CSF
predictions. R1 cases are identically the original energy functional.

## Orbital response and checks

For a nondegenerate selected CI root z, the exact analytic orbital jets are

    E_a  = z^T H_a z
    E_ab = z^T H_ab z
           + 2 sum_{s != root}
             (z_s^T H_a z)(z_s^T H_b z)/(E_root - E_s).

The second term is implemented; coefficients are not held frozen. Diagonal
orbital jets reuse the tested GFN2 ROKS response primitives. Off-diagonal
quartic-integral derivatives are explicit analytic contractions. Independent
checks compare the HF limit with a fermionic Hamiltonian contracted by PySCF
FCI, verify singlet S^2, and compare four first/second finite-rotation
directions. A real-molecule nonlinear GFN2 jet check and an all-available-root
identity audit are separate required gates.

Newton residual searches preserve local coherent multi-CSF overlap >=0.90,
normalized fixed-target fidelity squared >1/2, sector norms >=0.7, and CI-root
selection margin >=0.05. Final orbital residual <=1e-9 and a resolved actual
curvature index are required. A blocked path is not accepted. Retained-parent
and freeze/release initializations keep exactly the same functional and
thresholds and preserve earlier attempt checkpoints.

## Nuclear response and constraints

Analytic nuclear jets now include the frozen co-rotating Lowdin-frame ERI
transport, AO metric square-root jets, charge/spin frame response, the CI
response, and orbital relaxation. Furan F-parent checks of two reoptimized
Hessian columns agree within 2.2e-8 Ha/bohr^2.

The naive extension to four individual squared orbital cosines has a
rank-deficient Jacobian at the furan anchor. Its dependent second derivative
is nonzero on the tangent space, so simply dropping a constraint is invalid.
The failed attempt is retained, not regularized into a success.

`multicsf_frame_response.py` instead fixes the FULL active cross-overlap
matrix. A common ambient metric-orthogonal rotation and independent spectator
rotations preserve this matrix exactly. SVD eliminates gauge redundancy in
this explicit parameterization. This is a stricter, DIFFERENT experimental
PES. At independently stationary anchors, the projected analytic curvature is
exact. Two columns of the furan full excited-total Hessian, including all
three ground-reference derivatives, agree within 2.3e-8 Ha/bohr^2. Completed
parent Hessians and ground checkpoints are reused. Away from the anchor the analytic curvature includes the BCH common/private
product-chart terms and the mixed spectator-gauge terms omitted by an active-
only orbital Hessian. At a displaced Furan geometry, with individual parent
residuals around 2.3e-4 Ha but a stationary joint frame, two nuclear columns
agree within 2.3e-8 Ha/bohr^2. The independent orbital check differentiates
the exact analytic path gradient to avoid cancellation of total energies.

## Rank-rule controls and next candidate

The 95%-norm prescription improves five pi-pi states to MAE 0.1622 eV, but
lowers the aza-naphthalene n-pi prediction to 2.8389 eV against 5.346 eV. It is
rejected as a universal prescription. The Benzoquinone rank-two attempt has
unresolved curvature; its checkpoint and failure remain in the archive.

A separate proposal uses the inverse purity of normalized NTO weights:
Npair = 1/sum(w_i**2), r=floor(Npair+0.5). It uses no energy, family label,
fitted cutoff or interaction multiplier. On the fixed transitions it selects
rank two only for Cyclopentadiene, Furan and Phthalimide; all eleven other
vertical values reuse the identical original rank-one functional. Existing
same-rank converged parents suffice. MAEs are 0.2280 eV overall and 0.2138 eV
for pi-pi, versus matched TDA-Q 0.2612 and 0.4169 eV. This rule was proposed
AFTER examining the 95%-trial failure and all labels were already exposed;
it is development, not independent validation. Phthalimide Npair=1.5949 is
only 0.0949 above the rank boundary. Sensitivity and a fresh frozen validation
are still required. Rydberg accuracy is not corrected.

## Current limits

The candidate now has an explicit opt-in public selector, multicsf_50_50,
backed by portable checksummed frozen models and coherent APOC expansions.
It remains experimental and does not change the default method.
The original single-CSF method remains separate. LINK/APOC cold, displaced, relocated and resumed evaluations now pass; trial
points do not advance the accepted state and corrupt inputs are rejected. All
27 furan total-energy Hessian columns pass within 2.29e-8 Ha/bohr^2. Further
molecular column checks, actual DFT-anchor recalibration, independent chemical
validation and physical structure/frequency validation remain open. No broad accuracy or general timing advantage is established.
