# Analytic OO-DFT and ROKS nuclear response

Let `E = sum_t w_t E_t[D_t]`. DeltaSCF has one unrestricted determinant, with
independent alpha/beta orbital rotations. ROKS has weights `(2,-1)` for the
mixed Ms=0 and high-spin triplet determinants, with **one shared orthonormal
spatial orbital matrix** C. Their occupations remain binary and fixed.

For nonredundant rotations `C -> C exp(K)`, the orbital gradient is

`g_ij = 2 sum_(t,sigma) w_t (o_(t,sigma,j)-o_(t,sigma,i)) (C^T F_(t,sigma) C)_ij`.

Differentiating the Fock matrices with PySCF's analytic density-response kernel
and differentiating both orbital factors gives the complete moving-frame
Jacobian J. Newton–Krylov solves `J delta_K = -g`, including negative curvature.
The residual line search does not impose energy descent. A coherent singlet
CSF overlap guards the ROKS target. This root search does not impose a fixed
orbital saddle index or claim uniqueness.

At stationarity the total AO energy-weighted density is

`W = sym sum_(t,sigma) w_t S^-1 F_(t,sigma) D_(t,sigma)`.

It generally has off-diagonal orbital multipliers. Component canonical orbital
energies cannot replace it: the two ROKS determinants are not independently
stationary. The analytic gradient is

`G_x = sum_t w_t E_(t,x)|D - Tr(W S_x)`.

For a nuclear perturbation y, use the orthonormality connection
`C_y = C [K_y - (C^T S_y C)/2]`. Differentiating g at zero K gives B_y; the common
ROKS response solves `J K_y = -B_y`. DeltaSCF uses its independent spin rotations
in the same construction. No absolute-value Hessian or positive-curvature
replacement enters this CP equation.

With the resulting D_y, the Fock derivative is the sum of the explicit nuclear
Fock derivative and the density response, `F_y = F_y|D + kernel[D_y]`. The
energy-weighted density derivative follows from the product rule, including
`(S^-1)_y = -S^-1 S_y S^-1`. The nuclear Hessian is

`H_xy = sum_t w_t [E_(t,xy)|D + sum_sigma Tr(F_(t,sigma,x)|D D_(t,sigma,y))]`

`        - Tr(W S_xy) - Tr(W_y S_x)`.

PySCF's fixed-density nuclear integral derivatives supply the one-electron,
Coulomb and exchange contributions, including range separation. Setting the
component orbital energies to zero removes their diagonal-multiplier overlap
term; the full total W supplies the correct overlap term instead. The CP
residual and raw Hessian symmetry are checked before returning an array.

## Moving molecular quadrature

For an atom-centered grid, both its coordinates and Becke partition weights
move with the nuclei. The implementation adds an analytic moving-minus-fixed
XC correction to the native fixed-grid partial Hessian and explicit Fock
nuclear derivatives. This preserves the existing Coulomb/exchange and
range-separated derivative machinery.

For XC energy density f(rho) and a grid weight w, the full second derivative is

`(w f)_xy = w_xy f + w_x v.rho_y + w_y v.rho_x`

`            + w [rho_x.fxc.rho_y + v.rho_xy]`.

Here rho includes both spin densities and, for GGA, their three spatial-gradient
components. Second-order forward differentiation supplies `w_x,w_xy` from the
Becke partition, with the selected fixed atomic radii. Gaussian AO derivatives
through third order supply `rho_x,rho_xy` on moving grids. LibXC supplies vxc and
fxc. The same chain rule gives the nuclear Fock derivatives needed by the CP
equation. The code does not approximate these derivatives with finite steps,
freeze the grids, omit their response or reoptimize orbitals at displaced points.

## Spin and validation scope

The ROKS wavefunction is the coherent common-orbital two-determinant singlet,
not the auxiliary broken-symmetry determinant. Its spin-raising norm is checked
independently with fermionic bitstring operators. The functional energy `2 E_M
- E_T` remains an approximation; no exact molecular DFT projection theorem is
claimed. There is no fitted energy offset or semiempirical kernel scaling.

Executable remote checks and complete numerical certificates are archived under
`qualification/dft_roks_response_20261102`. They compare analytic Hessian–vector
products with Richardson-extrapolated differences of independently reconverged
analytic gradients. Finite differences are confined to those validation drivers.

## ROKS stationarity search and retained state

The moving orbital frame makes the residual Jacobian J nonsymmetric away from
stationarity. If Q is the orbital multiplier matrix and K(v) the antisymmetric
rotation, the connection correction is the selected antisymmetric part of
`K(v) Q - Q K(v)`. Consequently `J^T v = J v + connection(v)` and the symmetric
energy curvature is `J v + connection(v)/2`. The adjoint is used only for the
numerical residual search; the physical CP equation still uses J.

The default ROKS solver minimizes the full linearized residual inside a trust
ball using the span of a preconditioned Newton direction, `-J^T g`, and the
previous accepted direction. A thin SVD solves this small least-squares problem.
No orbital coordinates are removed from the energy or response. Actual versus
predicted residual reduction controls acceptance and the radius. Rejected trial
energies and Focks do not build XC response kernels; an accepted evaluation is
reused. GMRES uses a signed orbital-gap diagonal, so negative curvature is not
replaced by positive curvature.

`reference_coefficients` separates the physical target from a perturbed initial
guess. Step and target CSF overlaps guard continuity; neither proves a state
assignment. When the solver stalls, `freeze_and_release=True` prepares the
closed/virtual sector at fixed open orbitals and then releases the complete
orbital tangent. `initialize_closed_orbitals=True` requests this preparation
before a search without discarding a restart. These steps change initialization
only. A target must still be checked by orbital/transition character.

`certify_index=True` counts negative stationary orbital modes. For large tangents
an adaptive matrix-free lowest-eigenvalue prefix is expanded until a positive
boundary is found. `target_saddle_index` rejects a different index; it does not
steer the search toward the requested index. Checkpoints retain accepted
orbitals, immutable target, solver status, trust radius and iteration history.
A stalled tiny radius is reset on restart. Energy fingerprints remain unchanged.

Remote certificates in `qualification/roks_improvement_20261102` include an exact
adjoint comparison and perturbed retained-target recovery. The latter uses two
small molecules with three perturbations each; recovery can reuse a previously
stationary target. It tests numerical recovery, not discovery of six new states,
spectral accuracy, or superiority to MOM. Functional comparisons are a separate
historically exposed pilot and must not be described as blind validation.

Range-separated hybrids additionally reuse exact packed AO ERIs at each omega.
Mixed and triplet components share these density-independent integrals. The
cache is capped at one quarter of the configured mean-field memory budget;
unsupported/custom dispatch or insufficient budget uses the native exact J/K
path. No integral truncation or density fitting is introduced. The cache is
local to one orbital surface and its molecular geometry. Solver certificates
report allocated integral bytes, builds and hits. GMRES also uses an absolute
linear residual target of 0.1 times the nonlinear orbital tolerance, preventing
unnecessary oversolving near a stationary state; the final nonlinear residual
criterion is unchanged.
