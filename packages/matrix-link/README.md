
## Opt-in multi-CSF OO-DFT

Set `native_excited_method="multicsf_50_50"` and
`native_multicsf_model_path=Path(".../model.json")` with an explicit singlet
state and APOC tracking. Promotions, rank and full relative active frame are
stored in the model. See `examples/multicsf_oo_dft.py`.

Direct scans, geometry optimization and capped components use the same native
surface and analytic Hessian. An accepted generation saves both orbital sets,
CI coefficients, three ground references, derivatives and model fingerprint.
Only `accept` publishes it atomically; a trial or failed write leaves the
previous accepted state intact. Restart validates checksums and stationarity.
Model inputs are checked before optimizer cache hits. Moving the whole model
directory preserves its relative input references and accepted-state identity.

The furan cold/resumed test reproduces energy and Hessian, follows a displaced
point with coherent parent overlaps above 0.99998, and rejects corrupt inputs.
Surface evaluation performs no SCC, DFT or TDA calculation. Preparation costs
and independent chemical validation remain separate. Existing defaults remain
unchanged.
