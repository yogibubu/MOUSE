# Standard range-separated reference experiment

This exposed-development experiment retains lambda=0.9, common F/Q weights
0.5/0.5 and zero energy offsets. CAM-B3LYP uses its published short-/long-range
exchange fractions 0.19/0.65 and omega=0.33 bohr^-1. Long-range ERIs are
computed and cached once; the charge/spin kernel and independent full-AO Fock
checks include both exchange components. This gives an orbital-extent-dependent
exchange through integrals, without fitting an orbital-size parameter to TBEs.

Every Q reference uses the coherent full-basis CAM-B3LYP ground orbital levels.
The prepared native model, embeddings and target character are reused. All
nine states are exposed development, because previous functional tests had
already used their numerical references. The protocol retains every state,
unmodified accuracy gates and every completed checkpoint. The inherited log
phrase "coherent M06 reference" is a driver message, not the functional used;
the electronic functional and model coefficients are explicitly CAM-B3LYP.

This is a research candidate pending complete scoring and derivative validation.
The following native-orbital reference experiment tests the user's requested
separation of GFN2 spatial orbitals from DFT orbital-energy references, using
the same prepared response without additional DFT calculations.
