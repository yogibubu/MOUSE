The spectral validation workflow is molecule independent. Molecular identity,
monitored roots, CI windows, energy tolerance, successive increments and artifact
prefix are supplied in a JSON protocol. No molecule-specific energy corrections
or selection rules are introduced by these drivers.

Run only on the verified compute host, from the prepared qualification directory:

```sh
python complete_pigment_cid_validation.py --protocol chlorophyllide_cid_protocol.json
python audit_pigment_cid_validation.py --protocol chlorophyllide_cid_protocol.json
python collect_pigment_cid_validation.py --protocol chlorophyllide_cid_protocol.json --output chlorophyllide_cid_evidence_20261102
```

The first command resumes certified screening, scalar couplings and CI matrices.
If `wait_for_existing_baseline` is true, a separately launched baseline worker must
own the initial window. Otherwise, the driver starts that window itself. The
prepared full-quadrature context is cached across CI windows to avoid repeatedly
loading and checking identical data. Each window replaces its density span;
energies, quadrature, screening and physical model are unchanged.

Numerical spectral convergence requires both the coherent W=N sequence and the
fixed-interaction sequence to pass the frozen energy threshold. The audit checks
saved generalized eigenvectors independently, including the projected residual,
metric normalization, ground orthogonality, oscillator reconstruction, native
target coverage, root/subspace identities and dielectric algebra. Optical changes
are reported separately; energy convergence alone does not certify oscillator
strength convergence or agreement with experimental solution absorption maxima.

Large matrices remain on the compute host with checksums. Compact eigenvectors,
all computed root energies/intensities, configuration records, scalar journals,
source snapshots and the four ROKS checkpoints are archived for review. This
workflow establishes numerical properties of the prepared vertical model; it
makes no new nuclear-gradient/Hessian or blind-accuracy claim.
