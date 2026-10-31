# DFT versus GFN2 orbital relaxation: uracil and thymine

Oracle-only PBE0/aug-cc-pVDZ, identical cached geometries and ground references. Variational common-orbital ROKS objective 2 E(mixed) − E(triplet); all AO orbitals, including cores, can relax. The two-determinant CSF is a pure singlet. The functional energy is the approximate ROKS expression, not exact spin projection of DFT.

Completed and checked: 4/4. No fitted parameters. Stationarity uses an analytic orbital gradient and UKS response Hessian-vector products; directional finite differences check those derivatives. Newton-GMRES permits saddles. The Hessian index is not certified. This is not a nuclear Hessian of the hybrid model.

Thymine n→π* required a Cs-preserving variational initializer resumed from the retained Newton checkpoint. The geometry was not changed. The final Newton residual includes every active rotation, including those between different irreps. The initial failed derivative checks and subsequent finite-difference step refinements are retained; the original acceptance tolerances were unchanged.

| Molecule | State | DFT frozen | DFT relaxation | SE relaxation | DFT relaxed | Hybrid | QUEST |
|---|---|---:|---:|---:|---:|---:|---:|
| uracil | n_pi | 7.118453 | -2.393847 | -1.954226 | 4.724606 | 5.164227 | 4.935000 |
| uracil | pi_pi | 5.443767 | -0.421514 | -0.419531 | 5.022253 | 5.024236 | 5.381000 |
| thymine | n_pi | 6.938853 | -2.161526 | -1.689603 | 4.777327 | 5.249251 | 4.980000 |
| thymine | pi_pi | 5.273739 | -0.414814 | -0.421852 | 4.858925 | 4.851887 | 5.279000 |

All energies in eV. Hybrid = DFT frozen + SE relaxation.
Hybrid error − full DFT ROKS error = SE relaxation − DFT relaxation; this identity is independently checked against the saved energies.

| Molecule | State | Full DFT ROKS error | Relaxation transfer error | Hybrid error |
|---|---|---:|---:|---:|
| uracil | n_pi | -0.210394 | 0.439620 | 0.229227 |
| uracil | pi_pi | -0.358747 | 0.001983 | -0.356764 |
| thymine | n_pi | -0.202673 | 0.471924 | 0.269251 |
| thymine | pi_pi | -0.420075 | -0.007038 | -0.427113 |

| Molecule | QUEST π−n | DFT π−n | Hybrid π−n | DFT Δrelax(π−n) | SE Δrelax(π−n) |
|---|---:|---:|---:|---:|---:|
| uracil | 0.446000 | 0.297646 | -0.139991 | 1.972333 | 1.534696 |
| thymine | 0.299000 | 0.081598 | -0.397364 | 1.746713 | 1.267751 |

## Interpretation

Full PBE0 ROKS restores the n→π* / π→π* ordering for both molecules, although the splittings remain smaller than QUEST. For n→π*, GFN2 supplies too little relaxation: replacing it with the DFT relaxation lowers the hybrid energy by about 0.44 eV in uracil and 0.47 eV in thymine. For π→π*, SE and DFT relaxations agree within 0.008 eV: the remaining 0.36–0.42 eV underestimation is already present in PBE0 ROKS.

A change to the relaxation transfer alone cannot remove the π→π* error. The next model must distinguish the relaxation-transfer defect for n→π* from the reference-functional defect for π→π*, and be evaluated on the broader molecular set before any new orbital-dependent scaling is adopted. These four diagnostic cases do not establish a universal correction.

These four states are diagnostic cases selected after earlier errors, not an independent validation set. No timing claim against TDA is made. The hybrid nuclear gradient and analytic nuclear Hessian remain unavailable.
