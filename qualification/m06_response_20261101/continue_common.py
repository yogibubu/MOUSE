"""Ground-orthogonal stationary continuation from certified PBE0 to M06-2X."""
from predict import *
from native_batch import install_native_batch

PREVIOUS=Path('/home/enzo/MOUSE-stationary-anchor-20261101')

def run_continuation(cohort,name):
    out=DEST/cohort/name;case=prepare(cohort,name);install_native_batch(case,out)
    s=case['s'];no=case['no'];previous=PREVIOUS/cohort/name/'common_functional'
    paths=[previous/'model.npz',out/'common_functional/model.npz'];data=[]
    for path in paths:
        with np.load(path) as z:data.append({k:z[k].copy() for k in ['operators','charge','spin','q0','linear']})
    old,new=data;assert np.array_equal(old['operators'],new['operators'])
    oldlinear=old['linear']+old['charge']@(new['q0']-old['q0'])
    def at(t):
        folder=out/'continuation_models'/f't_{t:.8f}';folder.mkdir(parents=True,exist_ok=True);model=folder/'model.npz'
        if not model.exists():np.savez_compressed(model,operators=new['operators'],q0=new['q0'],
            charge=(1-t)*old['charge']+t*new['charge'],spin=(1-t)*old['spin']+t*new['spin'],linear=(1-t)*oldlinear+t*new['linear'])
        with np.load(model) as z:d={k:z[k].copy() for k in old}
        r=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,d,sha(model))
        seed=replace(loadpoint(previous/'ground/stationary.npz'),spin_reference_fingerprint=r.fingerprint)
        g=ground(folder/'ground',case['calc'],case['xyz'],s,r,seed=seed)
        context=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=r)
        return g,context
    rows=[dict(row,cohort=cohort) for row in json.loads((OUT/cohort/'manifest.json').read_text())['states'] if row['molecule']==name]
    for row in rows:
        dest=out/'continued_states'/row['id'];dest.mkdir(parents=True,exist_ok=True)
        if (dest/'result.json').exists() and json.loads((dest/'result.json').read_text())['qualified']:continue
        previous_state=PREVIOUS/cohort/name/'common_states'/row['id'];report=json.loads((previous_state/'result.json').read_text());assert report['qualified']
        source=previous_state/('stationary_from_F' if report.get('warm_start_checkpoint') else 'stationary')/'final.npz'
        with np.load(source) as z:c=z['coefficients'].copy()
        checkpoint=dest/'accepted.npz';current=0.;step=.125
        if checkpoint.exists():
            with np.load(checkpoint) as z:c=z['coefficients'].copy();current=float(z['t'])
        trace=json.loads((dest/'continuation.json').read_text()) if (dest/'continuation.json').exists() else []
        transition=target_transition(case,row);last=None
        for attempt in range(96):
            if current>=1-1e-12:break
            target=min(1.,current+step);g,context=at(target);cg=g.coefficients[0]
            q=cg[:,:no].T@s.overlap@transition@s.overlap@cg[:,no:];q/=np.linalg.norm(q)
            trial=dest/f'from_{current:.8f}_to_{target:.8f}';result=solve_stationary(context,case['eri'],no,c,cg,q,trial)
            trace.append(dict(t_from=current,t_target=target,kkt_max=result['kkt_max'],fidelity=result['fidelity'],qualified=result['qualified']))
            write(dest/'continuation.json',trace)
            if not result['qualified']:
                step*=.5
                if step<1/256:break
                continue
            with np.load(trial/'final.npz') as z:c=z['coefficients'].copy()
            current=target;last=result
            np.savez_compressed(checkpoint,t=current,coefficients=c)
            print(name,row['id'],'M06 continuation',current,flush=True)
        qualified=current>=1-1e-12
        result=dict(id=row['id'],molecule=name,qualified=qualified,reached_t=current,energy_labels_used=False,
            protocol_sha256=sha(DEST/'continuation_protocol.json'),source_sha256=sha(__file__),model_source_hashes=[sha(path) for path in paths])
        if qualified:
            g,context=at(1.);cg=g.coefficients[0];q=cg[:,:no].T@s.overlap@transition@s.overlap@cg[:,no:];q/=np.linalg.norm(q)
            final=solve_stationary(context,case['eri'],no,c,cg,q,dest/'final');result.update(final)
            if final['qualified']:result['excitation_ev']=(final['energy_hartree']-g.energy_hartree)*EV
        write(dest/'result.json',result)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        run_continuation('independent','Criegee_Intermediate')
        run_continuation('independent_challenge','Furofuran')
