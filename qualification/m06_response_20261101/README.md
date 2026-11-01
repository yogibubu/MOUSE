# M06-2X stationary-reference experiment

All parameters are fixed literature M06-2X values; lambda remains 0.9 in the
exchange response, F/Q weights 0.5/0.5, and excitation offsets zero. The full
meta-GGA response includes kinetic-energy density and all cross derivatives.
Exact unrestricted finite-difference Focks independently verify the assembled
charge and spin kernels to less than 5e-10 hartree on both initial molecules.

Changing the functional requires a coherent orbital-level reference. The
initial PBE0 spectral-reference variant is preserved, but is not the final
M06-2X variant. New full-basis M06-2X ground references supply coherent Q
spacings. Stationary continuation recovers the difficult Criegee pi-pi state
without changing its character or KKT thresholds. Completed SCF and orbital
checkpoints are resumed throughout.

Ground-projected spin-pure state interaction reduces the two difficult pi-pi
errors substantially. On four exposed development states plus two model-heldout
states the MAE is 0.236103 eV and maximum absolute error 0.394459 eV. These are
not blind reference data: older models had used the same molecules and labels.
The subsequent frozen acetaldehyde/formamide/pyrazine n-pi extension must be
included in judging this candidate, and exposes further transferability limits.
Do not promote or quote the six-state score as its complete validation.

The early diagnostic TDA calls inherited a 0.225 exchange coefficient from
the PBE0 helper. This does not affect ROKS or NOSI energies. The corrected
driver explicitly passes 0.486 for M06-2X; earlier embedded TDA fields are
superseded and must not be used as qualified M06-2X comparisons. Legacy flat
TDA amplitude arrays are reshaped according to their full occupied/virtual
dimensions; no target reference energy is used for assignments.

The model-independent complete analytic nuclear response is developed in
`../nosi_20261101`. Its molecular qualification is distinct from the energy
score. Large quadrature/response caches remain on Oracle.
