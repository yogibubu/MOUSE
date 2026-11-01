"""Prepare self-contained MOUSE OO-CID inputs; no TDA data in runtime bundle."""
from basis import *
from matrix_gfn2.spin_reference import save_response_reference
from matrix_gfn2.deltascf_derivatives import save_derivative_point
ROOT=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-response-consistent-nosi-20261101')

def export(cohort,name):
    folder=ROOT/'prepared_models/selected_cid'/cohort/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'model.json').exists():return folder/'model.json'
    d=prepare_basis(cohort,name);source=d['source'];case=d['case'];s=case['s'];result=ROOT/'selected_cid'/cohort/name
    report=json.loads((result/'result.json').read_text());ids=[r['id'] for r in report['primary']['predictions']];rows={row['id']:row for row in report['primary']['predictions']}
    with np.load(source/'coherent_common/model.npz') as z:data={k:z[k].copy() for k in ['operators','charge','spin','q0','linear']}
    response=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,data,sha(source/'coherent_common/model.npz'))
    if not (folder/'response.npz').exists():save_response_reference(folder/'response.npz',response)
    if not (folder/'template.npz').exists():save_derivative_point(folder/'template.npz',replace(case['raw'],reference_state=s),request_hash='nosi:prepared:template',coordinates_angstrom=case['xyz'])
    if not (folder/'arrays.npz').exists():
        with np.load(source/'coherent_common/ground/stationary.npz') as z:cg=z['result_coefficients'][0].copy()
        states=[];mu=[]
        for identity in ids:
            state=source/'coherent_states'/identity
            with np.load(state/'final/final.npz') as z:states.append(z['coefficients'].copy())
            mu.append(json.loads((state/'result.json').read_text())['multiplier'])
        with np.load(result/'checkpoint.npz') as z:ci=z['coefficients'][:,[rows[identity]['root'] for identity in ids]].copy()
        temp=folder/'arrays.tmp.npz';np.savez_compressed(temp,h=d['h'],eri=d['eri'],ground=cg,states=states,multipliers=mu,auxiliary_frame=d['sr']@d['canonical'],reference_ci=ci);temp.replace(folder/'arrays.npz')
    write(folder/'model.json',dict(schema='mouse.nosi.selected_cid.v1',weights=[.5,.5],lambda_exchange=.9,energy_offset_ev=0,state_ids=ids,labels=report['labels'],sha256={name:sha(folder/name) for name in ['template.npz','response.npz','arrays.npz']},provenance=dict(candidate_freeze_commit='467a9e5',native_model_sha256=sha(source/'coherent_common/model.npz'),selected_matrix_sha256=sha(result/'checkpoint.npz'),runtime_DFT_backend_required=False,runtime_TDA_required=False,source_sha256=sha(__file__))))
    return folder/'model.json'
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');args=p.parse_args()
    with threadpool_limits():print(export(args.cohort,args.name),flush=True)
