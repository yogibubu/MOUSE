"""Fully shared-orbital limit: optimize one exact 50/50 energy functional."""
from predict import *
from native_batch import install_native_batch
from matrix_gfn2.multicsf import MultiCSFObjective

def run(cohort,name):
    out=DEST/cohort/name;case=prepare(cohort,name);install_native_batch(case,out);s=case['s'];no=case['no']
    case['calc'].evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC forbidden after preparation'))
    scf.hf.kernel=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('DFT SCF forbidden after preparation'))
    data=[];contexts=[];grounds=[]
    for profile in ['F','Q']:
        with np.load(out/profile/'model.npz') as a:d={k:a[k].copy() for k in ['operators','charge','spin','q0','linear']}
        data.append(d);r=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,d,sha(out/profile/'model.npz'))
        contexts.append(prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=r))
        grounds.append(ground(out/profile/'ground',case['calc'],case['xyz'],s,r))
    for key in ['operators','charge','spin','q0']:assert np.array_equal(data[0][key],data[1][key])
    folder=out/'common_functional';folder.mkdir(exist_ok=True);model=folder/'model.npz'
    if not model.exists():np.savez_compressed(model,**dict(data[0],linear=.5*(data[0]['linear']+data[1]['linear'])))
    with np.load(model) as a:d={k:a[k].copy() for k in data[0]}
    r=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,d,sha(model))
    context=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=r)
    seed=replace(grounds[0].reference,spin_reference_fingerprint=r.fingerprint)
    g=ground(folder/'ground',case['calc'],case['xyz'],s,r,seed=seed);cg=g.coefficients[0]
    rows=[dict(row,cohort=cohort) for row in json.loads((OUT/cohort/'manifest.json').read_text())['states'] if row['molecule']==name]
    for row in rows:
        dest=out/'common_states'/row['id'];dest.mkdir(parents=True,exist_ok=True)
        retry=False
        if (dest/'result.json').exists():
            if json.loads((dest/'result.json').read_text())['qualified']:continue
            (dest/'result.json').replace(dest/'initial_NTO_attempt.json');retry=True
        transition=target_transition(case,row);q=cg[:,:no].T@s.overlap@transition@s.overlap@cg[:,no:];q/=np.linalg.norm(q)
        u,sv,vh=np.linalg.svd(q,full_matrices=True);initial=np.column_stack([cg[:,:no]@u,cg[:,no:]@vh.T])
        # Exact functional identity at the same orbitals, including analytic orbital curvature.
        originals=[MultiCSFObjective(ctx,case['eri'],no,1,cg,q).evaluate(initial,True) for ctx in contexts]
        averaged=MultiCSFObjective(context,case['eri'],no,1,cg,q).evaluate(initial,True)
        errors=[float(np.max(abs(value-.5*(old0+old1)))) for value,old0,old1 in zip(averaged,originals[0],originals[1])]
        assert max(errors)<1e-10,errors
        solver_folder=dest/'stationary';warm_source=None
        if retry:
            # Reuse a certified target of a neighboring parent Hamiltonian;
            # no energy label or orbital-character threshold is changed.
            for profile in ['F','Q']:
                certificate=out/'states'/row['id']/profile/'result.json'
                checkpoint=out/'states'/row['id']/profile/'final/final.npz'
                if certificate.exists() and checkpoint.exists() and json.loads(certificate.read_text())['qualified']:
                    with np.load(checkpoint) as a:initial=a['coefficients'].copy()
                    solver_folder=dest/f'stationary_from_{profile}';warm_source=str(checkpoint);break
        result=solve_stationary(context,case['eri'],no,initial,cg,q,solver_folder)
        result.update(id=row['id'],molecule=name,family=row['family'],energy_labels_used=False,
            weights=[.5,.5],lambda_exchange=.9,offset_ev=0,identity_energy_gradient_hessian_errors=errors,
            model_sha256=sha(model),driver_source_sha256=sha(__file__),protocol_sha256=sha(DEST/'common_protocol.json'),
            scope='one averaged energy functional, fully shared orbitals; different from averaging separately optimized parent energies',
            warm_start_checkpoint=warm_source)
        if result['qualified']:result.update(excitation_ev=(result['energy_hartree']-g.energy_hartree)*EV,tda=tda(context,g,case['eri'],s,no,transition))
        write(dest/'result.json',result);print(name,row['id'],'common 50/50 functional',result['qualified'],flush=True)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        run('independent','Criegee_Intermediate')
        run('independent_challenge','Furofuran')
