"""Compute only reference identities/native controls before the candidate guard."""
from remote_bootstrap import BASE,HOST
import json,sys
from pyscf import lib
from reference_td_checkpoint import reference
import prepare_E
lib.num_threads(4)
protocol=BASE/'edft_holdout_E_protocol.json';case=json.loads(protocol.read_text())['cases'][int(sys.argv[1])]
print('E PREP HOST',HOST,case['name'],flush=True)
reference(case,protocol);prepare_E.prepare(case)
