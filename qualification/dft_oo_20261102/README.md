# Initial true-DFT OO qualification

Run `validate.py` and `validate_extended.py` only on the verified MAC-STUDIO.
They load the adjacent copy of the public `matrix_pyscf/deltascf.py` source;
the exported source hash links that copy to the committed implementation.

Water/STO-3G checks HF and PBE0 orbital-response actions, nuclear gradients,
same-geometry restart and agreement with PySCF MOM. The additional
formaldehyde/6-31G PBE0 calculation checks a genuine negative-curvature stationary
point, an independent nuclear-gradient direction, APOC export and rejection of a
wrong-functional restart. Finite differences appear only in validation.

The certificates describe unrestricted determinants, with spin contamination
reported explicitly. They establish neither singlet spin purity, nuclear-Hessian
availability, experimental accuracy nor superiority over MOM. No pigment results
or semiempirical accuracy statistics are changed by these tests.
