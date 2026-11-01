# Closest fermionic-response diagnostic

This is an isolated, closed development diagnostic, not a replacement for OO
ROKS. It projects the stationary embedded charge and spin kernels onto a true
two-body fermionic Hamiltonian by the parameter-free formula in
`response_projection.py`. Normal ordering preserves the input mean F/Q Fock
at its reference density. Kernel residuals remain nonzero: this is a new
approximation, not an exact recasting of the DFT functional.

The frozen protocol uses one CAS(6,6) per molecule, comprising the union of
leading target NTOs and frontier extensions. The full 400-dimensional Ms=0
space is projected onto its exact 175-dimensional singlet space. Hamiltonian
Hermiticity and commutation with S-squared are checked. All molecular jobs
ran on oracle.sns.it with the shared compute lock, reusing the existing DFT
reference without new SCF calculations. Three algebraic tests passed there.

Neither molecule yields accepted assignments under the predeclared overlap
and margin thresholds. Criegee's largest normalized transition-character
overlaps occur at weak transitions to high roots, with insufficient margins.
Furofuran likewise fails the fixed assignment criteria. These energies are
diagnostic spectrum entries, not accepted predictions of the requested
states. No thresholds were relaxed and no orbital optimization or nuclear
derivative claim follows. This closes the proposed projection route at its
initial vertical diagnostic; it does not resolve the accuracy problem.

`remote/` contains the executed driver, protocol, output log, complete singlet
spectra and molecular checkpoints. The initial driver run stopped on a saved
Fock-field naming error before any molecular diagonalization; the corrected
driver uses `result_fock` and completes both molecules.
