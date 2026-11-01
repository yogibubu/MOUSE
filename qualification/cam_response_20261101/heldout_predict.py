"""Fixed candidate on the two additional molecules; deterministic seeds."""
from predict import *
from native_batch import install_native_batch

def predict_molecule(cohort,name):
    case=prepare(cohort,name);out=DEST/cohort/name;install_native_batch(case,out);s=case['s'];no=case['no']
    model=out/'coherent_common/model.npz'
    with np.load(model) as z:data={k:z[k].copy() for k in ['operators','charge','spin','q0','linear']}
    r=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,data,sha(model))
    ctx=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=r)
    with np.load(out/'embedded_ground.npz') as z:c=z['coefficients'].copy()
    p=c[:,:no]@c[:,:no].T;f,e=ctx.fock_and_energy(np.array([p,p]))
    seed=replace(case['raw'],coefficients=np.array([c,c]),fock=f,energy_hartree=e,spin_reference_fingerprint=r.fingerprint,reference_state=s,converged=True)
    g=ground(out/'coherent_common/ground',case['calc'],case['xyz'],s,r,seed=seed);cg=g.coefficients[0]
    rows=[dict(row,cohort=cohort) for row in json.loads((OUT/cohort/'manifest.json').read_text())['states'] if row['molecule']==name]
    for row in rows:
        dest=out/'coherent_states'/row['id'];dest.mkdir(parents=True,exist_ok=True)
        if (dest/'result.json').exists() and json.loads((dest/'result.json').read_text())['qualified']:continue
        transition=target_transition(case,row);q=cg[:,:no].T@s.overlap@transition@s.overlap@cg[:,no:];q/=np.linalg.norm(q)
        u,sv,vh=np.linalg.svd(q,full_matrices=True);initial=np.column_stack([cg[:,:no]@u,cg[:,no:]@vh.T])
        candidates=[('NTO',initial)]
        previous=case.get('retained_seed',Path('/home/enzo/MOUSE-transferability-20261101/stationary_analytic/orthogonal')/cohort/name/row['id']/'parent_0/final.npz')
        if previous.exists():
            with np.load(previous) as z:candidates.append(('retained_F',z['coefficients'].copy()))
        accepted=False;attempts=[]
        for label,initial in candidates:
            result=solve_stationary(ctx,case['eri'],no,initial,cg,q,dest/label)
            attempts.append(dict(seed=label,qualified=result['qualified'],kkt_max=result['kkt_max'],fidelity=result['fidelity']))
            if result['qualified']:
                with np.load(dest/label/'final.npz') as z:c=z['coefficients'].copy()
                accepted=True;break
        # Deterministic continuation from a previously certified parent, with
        # the same acceptance gates; no target energy is consulted.
        if not accepted and previous.exists():
            with np.load(previous) as z:c=z['coefficients'].copy()
            oldpath=case['src']/'full_response/anchor/model.npz'
            with np.load(oldpath) as z:old={k:z[k].copy() for k in data}
            linear=old['linear']+old['charge']@(data['q0']-old['q0']);current=0.;step=.125
            checkpoint=dest/'accepted.npz'
            if checkpoint.exists():
                with np.load(checkpoint) as z:c=z['coefficients'].copy();current=float(z['t'])
            for _ in range(96):
                if current>=1-1e-12:break
                target=min(1.,current+step);folder=out/'heldout_continuation_models'/f't_{target:.8f}';folder.mkdir(parents=True,exist_ok=True)
                path=folder/'model.npz'
                if not path.exists():np.savez_compressed(path,operators=data['operators'],q0=data['q0'],charge=(1-target)*old['charge']+target*data['charge'],spin=(1-target)*old['spin']+target*data['spin'],linear=(1-target)*linear+target*data['linear'])
                with np.load(path) as z:d={k:z[k].copy() for k in data}
                rr=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,d,sha(path))
                gg=ground(folder/'ground',case['calc'],case['xyz'],s,rr,seed=replace(seed,spin_reference_fingerprint=rr.fingerprint))
                cc=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=rr)
                reference=gg.coefficients[0];qq=reference[:,:no].T@s.overlap@transition@s.overlap@reference[:,no:];qq/=np.linalg.norm(qq)
                trial=dest/f'from_{current:.8f}_to_{target:.8f}';result=solve_stationary(cc,case['eri'],no,c,reference,qq,trial)
                attempts.append(dict(t_from=current,t_to=target,qualified=result['qualified'],kkt_max=result['kkt_max'],fidelity=result['fidelity']));write(dest/'attempts.json',attempts)
                if not result['qualified']:
                    step*=.5
                    if step<1/256:break
                    continue
                with np.load(trial/'final.npz') as z:c=z['coefficients'].copy()
                current=target;np.savez_compressed(checkpoint,t=current,coefficients=c)
            accepted=current>=1-1e-12
        if accepted:
            result=solve_stationary(ctx,case['eri'],no,c,cg,q,dest/'final')
            if result['qualified']:result['excitation_ev']=(result['energy_hartree']-g.energy_hartree)*EV
        else:result=dict(qualified=False)
        result.update(id=row['id'],molecule=name,attempts=attempts,energy_labels_used=False,source_sha256=sha(__file__),protocol_sha256=sha(DEST/'heldout_protocol.json'))
        write(dest/'result.json',result);print(name,row['id'],result['qualified'],flush=True)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        for name in ['Benzoxadiazole','Hexatriene']:predict_molecule('independent',name)
