"""Q-projected excited-space interaction with the stationary S0 fixed."""
from diagnostic import *

def projected(cohort,name):
    folder=HERE/cohort/name
    output=folder/'projected_result.json'
    if output.exists():return
    with np.load(folder/'checkpoint.npz') as z:s=z['metric'].copy();h=z['matrix'].copy()
    p=np.eye(len(s))[:,1:];p[0,:]-=s[0,1:]/s[0,0]
    metric=p.T@s@p;matrix=p.T@h@p
    w,u=eigh(metric);keep=w>1e-8;orth=u[:,keep]/np.sqrt(w[keep])
    e,z=eigh(orth.T@matrix@orth);z=p@orth@z;overlaps=abs(s@z)
    report=json.loads((folder/'result.json').read_text());rows=report['labels'][1:3]
    assigned,roots=linear_sum_assignment(-overlaps[1:3])
    predictions=[]
    for index,root in zip(assigned,roots):
        ov=float(overlaps[index+1,root]);margin=ov-float(np.max(np.delete(overlaps[index+1],root),initial=0.))
        predictions.append(dict(id=rows[index],root=int(root+1),gap_ev=float(e[root]*EV),overlap=ov,margin=margin,qualified=bool(ov>=.7 and margin>=.05)))
    assert np.max(abs(s[0]@z))<1e-10
    np.savez_compressed(folder/'projected_checkpoint.npz',metric=metric,matrix=matrix,coefficients=z,energies=e,overlaps=overlaps)
    write(output,dict(predictions=predictions,host=HOST,all_gaps_ev=e*EV,ground_overlap_max=float(np.max(abs(s[0]@z))),
        energy_labels_used=False,offset_ev=0,weights=[.5,.5],lambda_exchange=.9,
        protocol_sha256=sha(HERE/'projected_protocol.json'),source_sha256=sha(__file__),base_checkpoint_sha256=sha(folder/'checkpoint.npz')))
    print(name,predictions,flush=True)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        projected('independent','Criegee_Intermediate')
        projected('independent_challenge','Furofuran')
