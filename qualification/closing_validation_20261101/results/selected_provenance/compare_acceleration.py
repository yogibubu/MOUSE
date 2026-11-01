"""Exact same-model public-result comparison, with archived serial timing."""
from pathlib import Path
import socket,runpy,json
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';OLD=BASE/'oracle/MOUSE-pi-pi-transfer-repair-20261101'
runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
import numpy as np
from matrix_gfn2.nosi_surface import PreparedNOSIModel,load_nosi_result
m=PreparedNOSIModel(OLD/'prepared_models/independent/Criegee_Intermediate/model.json')
a=load_nosi_result(OLD/'portable_validation/independent/Criegee_Intermediate/checkpoint.npz',m);b=load_nosi_result(HERE/'acceleration/independent/Criegee_Intermediate/checkpoint.npz',m)
errors={}
for x,y in zip(a.states,b.states):
 errors[x.state_id]=dict(energy_error_hartree=abs(x.energy_hartree-y.energy_hartree),gradient_error=float(np.max(abs(x.gradient_hartree_per_bohr-y.gradient_hartree_per_bohr))),hessian_error=float(np.max(abs(x.hessian_hartree_per_bohr2-y.hessian_hartree_per_bohr2))))
 assert max(errors[x.state_id].values())<1e-11,errors
old=json.loads((OLD/'portable_validation/independent/Criegee_Intermediate/hessian.json').read_text());new=json.loads((HERE/'acceleration/independent/Criegee_Intermediate/hessian.json').read_text());report=dict(errors=errors,before_seconds=old['wall_seconds'],after_seconds=new['wall_seconds'],ratio=old['wall_seconds']/new['wall_seconds'],percent_reduction=100*(1-new['wall_seconds']/old['wall_seconds']),before_peak_rss_bytes=old['peak_rss_native'],after_peak_rss_bytes=new['peak_rss_native'],scope='one cold two-thread serial public-API run per version; other host tasks active, modest difference not a robust scaling claim',changes='vectorized exact residual matrix completion; direct frozen auxiliary values without scalar jet allocation',model_changed=False)
(HERE/'acceleration/comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
