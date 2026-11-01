"""Export the frozen screened-even CID as a self-contained ground-engine model."""
from pathlib import Path
import sys,json,socket
assert socket.getfqdn()=='oracle.sns.it'
ROOT=Path('/home/enzo/MOUSE-response-consistent-nosi-20261101');HERE=Path('/home/enzo/MOUSE-pi-pi-transfer-repair-20261101')
sys.path.insert(0,str(ROOT))
from basis import *
from matrix_gfn2.spin_reference import save_response_reference
from matrix_gfn2.deltascf_derivatives import save_derivative_point

def export(item):
 cohort,name=item['cohort'],item['molecule'];folder=HERE/'prepared_models'/cohort/name;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'model.json').exists():return
 d=prepare_basis(cohort,name);source=d['source'];case=d['case'];s=case['s'];result=HERE/'even_sector_5'/cohort/name;report=json.loads((result/'result.json').read_text());ids=[r['id'] for r in report['primary']['predictions']];rows={r['id']:r for r in report['primary']['predictions']}
 with np.load(source/'coherent_common/model.npz') as z:data={k:z[k].copy() for k in z.files}
 response=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,data,sha(source/'coherent_common/model.npz'))
 save_response_reference(folder/'response.npz',response);save_derivative_point(folder/'template.npz',replace(case['raw'],reference_state=s),request_hash='nosi:prepared:template',coordinates_angstrom=case['xyz'])
 states=[];multipliers=[]
 for identity in ids:
  state=source/'coherent_states'/identity
  with np.load(state/'final/final.npz') as z:states.append(z['coefficients'].copy())
  multipliers.append(json.loads((state/'result.json').read_text())['multiplier'])
 op=HERE/'expanded_cid_fast_5'/cohort/name/'operators.npz'
 if not op.exists():op=HERE/'expanded_cid_5'/cohort/name/'operators.npz'
 with np.load(op) as z:h=z['h'].copy();w=z['eri'].copy()
 with np.load(result/'checkpoint.npz') as z:ci=z['coefficients'][:,[rows[identity]['root'] for identity in ids]].copy()
 np.savez_compressed(folder/'arrays.npz',h=h,eri=w,ground=d['cg'],states=states,multipliers=multipliers,auxiliary_frame=d['sr']@d['canonical'],reference_ci=ci)
 write(folder/'model.json',dict(schema='mouse.nosi.screened_even_cid.v1',active_space_size=5,interaction='static_constrained_RPA',augmentation='frozen_reflection_even',weights=[.5,.5],lambda_exchange=.9,energy_offset_ev=0,state_ids=ids,labels=report['labels'],sha256={file:sha(folder/file) for file in ['template.npz','response.npz','arrays.npz']},provenance=dict(candidate_freeze_commit='2d59ab8',native_model_sha256=sha(source/'coherent_common/model.npz'),selected_matrix_sha256=sha(result/'checkpoint.npz'),reflection_certificate=report['reflection'],runtime_DFT_backend_required=False,runtime_TDA_required=False,source_sha256=sha(__file__))))
 print(name,'exported',flush=True)
if __name__=='__main__':
 with threadpool_limits():
  for item in json.loads((HERE/'cases.json').read_text()):export(item)
