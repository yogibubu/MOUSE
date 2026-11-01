"""Reprepare the joint state frame on the new common stationary ground."""
from predict import *
from joint_projected import solve_joint_projected
from native_batch import install_native_batch
from matrix_gfn2.multicsf import MultiCSFObjective

def run(cohort,name):
    out=DEST/cohort/name;case=prepare(cohort,name);install_native_batch(case,out)
    s=case['s'];no=case['no'];objects_data=[]
    case['calc'].evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC forbidden after preparation'))
    scf.hf.kernel=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('DFT SCF forbidden after preparation'))
    for profile in ['F','Q']:
        with np.load(out/profile/'model.npz') as a:data={k:a[k].copy() for k in ['operators','charge','spin','q0','linear']}
        response=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,data,sha(out/profile/'model.npz'))
        g=ground(out/profile/'ground',case['calc'],case['xyz'],s,response)
        ctx=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=response)
        objects_data.append((ctx,g))
    rows=[dict(r,cohort=cohort) for r in json.loads((OUT/cohort/'manifest.json').read_text())['states'] if r['molecule']==name]
    for row in rows:
        dest=out/'reprepared_states'/row['id'];dest.mkdir(parents=True,exist_ok=True)
        if (dest/'result.json').exists():continue
        transition=target_transition(case,row);c0=objects_data[0][1].coefficients[0]
        q=c0[:,:no].T@s.overlap@transition@s.overlap@c0[:,no:];q/=np.linalg.norm(q)
        u,sv,vh=np.linalg.svd(q,full_matrices=True);initial=np.column_stack([c0[:,:no]@u,c0[:,no:]@vh.T])
        objects=[];grounds=[];caps=[]
        for ctx,g in objects_data:
            cg=g.coefficients[0];q=cg[:,:no].T@s.overlap@transition@s.overlap@cg[:,no:];q/=np.linalg.norm(q)
            caps.append(float(np.linalg.svd(q,compute_uv=False)[0]))
            objects.append(MultiCSFObjective(ctx,case['eri'],no,1,cg,q));grounds.append(g)
        np.savez_compressed(dest/'initial_frame.npz',coefficients=np.array([initial,initial]),active_cross_overlap=np.eye(2),transition_weights=sv**2)
        result,point=solve_joint_projected(objects,grounds,[initial.copy(),initial.copy()],dest/'stationary')
        result.update(id=row['id'],molecule=name,family=row['family'],rank_caps=caps,energy_labels_used=False,
            weights=[.5,.5],lambda_exchange=.9,offset_ev=0,preparation='common new-ground NTO active frame, not the old-model cross-parent Gram',
            driver_source_sha256=sha(__file__),protocol_sha256=sha(DEST/'reprepared_protocol.json'))
        if result['qualified']:
            gaps=[(e-g.energy_hartree)*EV for e,g in zip(result['parent_energies'],grounds)]
            result.update(parent_gaps_ev=gaps,excitation_ev=.5*sum(gaps),
                tda=[tda(ctx,g,case['eri'],s,no,transition) for ctx,g in objects_data])
        write(dest/'result.json',result);print(name,row['id'],'new-ground joint frame',result['qualified'],flush=True)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        run('independent','Criegee_Intermediate')
        run('independent_challenge','Furofuran')
