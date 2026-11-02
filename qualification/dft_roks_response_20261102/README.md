# Analytic full-DFT ROKS / DeltaSCF response qualification

All electronic and numerical work ran on the verified MAC-STUDIO. The public
API is `matrix_pyscf.optimize_roks(..., include_hessian=True)` or
`optimize_deltascf(..., include_hessian=True)`. No finite difference enters those
production routines. The drivers assert the compute host and use the existing
remote source bootstrap.

`validate.py` checks water/STO-3G with HF and PBE0. `validate_extended.py` adds
formaldehyde/6-31G PBE0, water/CAM-B3LYP, water/LDA,VWN and unrestricted
water/PBE0. Independent scalar-energy derivatives and Richardson differences of
reconverged gradients validate the analytic gradient and Hessian. ROKS CP is a
coupled common-orbital response; its mixed/triplet components are not separately
stationary. Coherent CSF tracking, restart and an explicit fermionic spin-raising
operator check the state representation.

The six checks cover two molecules, five ROKS surfaces and one unrestricted
DeltaSCF surface. DFT Hessian-vector errors are at most 5.02e-9 Ha/bohr². HF/water
has a soft response: its independent finite-step error is 2.42e-7, at the two
small steps documented in the certificate. These checks establish derivative
implementation consistency, not experimental spectral accuracy or transferability.
The constructed ROKS CSF is spin-pure; its common-orbital `2 E_M - E_T` functional
energy remains approximate. The unrestricted Ms=0 determinant is spin contaminated.

`replay_cached_response.py` checks all complete arrays after caching spin-density
AO products. The largest change is 2.38e-13 Ha/bohr²; translation sum-rule errors
are below 1.35e-11. Completed orbital checkpoints are reused with zero orbital
iterations and no new ground SCF. One-AO hydrogen checks the zero-rotation limit;
unsupported meta-GGA Hessians are rejected before electronic iteration.

`collect.py` exports source snapshots, arrays, logs, checkpoints, scalar reports
and a SHA-256 manifest. The `results/executed_sources` copies match the committed
public implementation. The earlier uncached grid source and original diagnostic
logs are retained separately. `results/summary.json` and
`results/cached_response_validation.json` are the compact entry points.

The supported nuclear-Hessian scope is full-integral, all-electron molecular
UHF and LDA/GGA UKS, including global/range-separated hybrids, with standard
original-Becke partition/radius adjustments. Density fitting, ECP Hessians,
meta-GGA Hessians, solvent and relativistic response require separate extensions.
