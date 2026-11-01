# Spin-pure state interaction and complete analytic response

This is isolated research code. It does not silently replace the production
relative-parent model. The CSF-level overlap-weighted residual functional is
motivated by MSDFT-NOSI (Zhu et al., doi:10.1021/acs.jpca.3c04799), but is a
distinct approximation. Both full and stationary-ground-projected spectra
are retained. No target excitation energy enters the model or root solver.

`slater_condon.py` computes singular-safe determinant matrix elements;
`elements_mixed.py` supplies analytic first and mixed second element jets.
A Schur complement isolates a tiny null-overlap block; its determinant and
cofactors are evaluated as division-free polynomials. This avoids differentiating
an inverse at singular occupied overlaps. Independent tests include nullities
zero through three and comparison with literal determinant matrix elements.

`mixed_fock.py` extends the native analytic molecular Fock algebra to the
complete frozen one-body/charge/spin correction, retaining metric and frame
terms with early feature contraction. `orbital_mixed.py` differentiates the
entire stationary-ground and ROKS KKT systems twice, including the moving
ground determinant, overlap constraint and its multiplier. It supplies the
orbital second response required by nonvariational state interaction.

`nuclear_assembly.py` assembles actual molecular gradients and nuclear Hessians,
including generalized eigenvector, metric and ground-projector response. The
physical Slater-Condon h/ERI operators are frozen in a co-rotating orthonormal
frame; native semiempirical diagonal energies keep their full geometry and
density dependence. This PES definition is explicit in `nuclear_protocol.json`.

The complete Criegee 15-coordinate calculation on Oracle takes 41.49 s for
both states in this implementation. Its raw Hessian symmetry error is below
7e-14 hartree/bohr² and the translational sum-rule residual below 1.4e-13.
Independent, tightly reconverged displaced roots verify both states. Central
differences show their expected quadratic step error; Richardson extrapolation
of the independent validation values gives maximum gradient and directional
curvature errors of 9.8e-9 hartree/bohr and 3.4e-7 hartree/bohr². The delivered
Hessian itself uses no derivative steps or displaced electronic calculations.

The Furofuran complete 36-coordinate Hessians have also been assembled on
the explicitly verified MAC-STUDIO, preserving the prepared Oracle inputs.
Independent displaced-energy validation is a separate stage. Native binaries
are built on the target architecture; only workspace paths and the explicit
host assertion are changed in the port, with a source-hash manifest retained.
On macOS `ru_maxrss` is bytes, whereas Linux reports KiB: the historical raw
certificate field `peak_rss_kib` must be interpreted with the host convention.

These response checks establish derivatives of the specified research model;
they do not establish its accuracy or transferability. The M06-2X candidate
passes the first six-state gate but its additional n-pi tests expose larger
errors. That functional is not promoted on the strength of the six-state
subset. Subsequent reference changes retain all previous protocols and data.

All molecular/electronic work is restricted to verified Oracle or MAC-STUDIO.
The laptop only edits, transfers and inspects data. Complete outputs and
checkpoints remain in the respective remote study roots.
