"""Guarded ground-orthogonal ROKS continuation to the stationary-anchor model."""
from anchor import *
sys.path.insert(0,'/home/enzo/MOUSE-transferability-20261101')
from stationary_orthogonal_diagnostic import solve_stationary
from matrix_gfn2.relative_pair_roks import load_relative_pair_roks
from tda_operator import complete_tda_A

def response_and_ground(case,out,profile,t,seed):
    s=case['s'];i=['F','Q'].index(profile)
    final=out/profile/'model.npz'
    if t==1.:model=final;folder=out/profile
    else:
        folder=out/profile/'continuation'/f't_{t:.8f}';folder.mkdir(parents=True,exist_ok=True);model=folder/'model.npz'
        if not model.exists():
            oldpath=case['src']/['full_response','qualified_spectral_response'][i]/'anchor/model.npz'
            with np.load(oldpath) as f:old={k:f[k].copy() for k in ['operators','charge','spin','q0','linear']}
            with np.load(final) as f:new={k:f[k].copy() for k in old}
            assert np.array_equal(old['operators'],new['operators'])
            linear_old=old['linear']+old['charge']@(new['q0']-old['q0'])
            data=dict(operators=new['operators'],q0=new['q0'],charge=(1-t)*old['charge']+t*new['charge'],
                spin=(1-t)*old['spin']+t*new['spin'],linear=(1-t)*linear_old+t*new['linear'])
            np.savez_compressed(model,**data)
    with np.load(model) as f:data={k:f[k].copy() for k in ['operators','charge','spin','q0','linear']}
    response=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,data,sha(model))
    seed=replace(seed.reference if isinstance(seed,RestrictedGroundResult) else seed,spin_reference_fingerprint=response.fingerprint)
    g=ground(folder/'ground',case['calc'],case['xyz'],s,response,seed=seed)
    ctx=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=response)
    return response,g,ctx

def target_transition(case,row):
    s=case['s'];no=case['no'];small,transform=source_molecule(s)
    cross=(gto.intor_cross('int1e_ovlp',small,case['mol']).T@transform).T
    overlap=s.coefficients.T@cross@case['cd'];dn=np.count_nonzero(case['od']==2)
    with np.load(case['ref']/f'{row["id"]}.npz') as a:dx=a['amplitudes'][row['ordinal_same_symmetry']-1].copy()
    if dx.ndim==1:dx=dx.reshape(dn,case['cd'].shape[1]-dn)
    q=overlap[:no,:dn]@dx@overlap[no:,dn:].T;q/=np.linalg.norm(q)
    return s.coefficients[:,:no]@q@s.coefficients[:,no:].T

def tda(context,g,eri,s,no,transition):
    e,c=eigh(g.fock[0],s.overlap);state=SimpleNamespace(**vars(s));state.coefficients=c
    state.fock=g.fock[0];state.orbital_energies_hartree=e
    kernel=context.density_response_kernel(state)
    ii=np.repeat(np.arange(no),len(c)-no);aa=np.tile(np.arange(no,len(c)),no)
    ci=c[:,ii];ca=c[:,aa];columns=[]
    for i,a in zip(ii,aa):
        p=np.outer(c[:,i],c[:,a]);f=kernel(.5*(p+p.T));columns.append(np.einsum('mi,mn,ni->i',ci,f,ca))
    static=np.diag(e[aa]-e[ii])+2*np.array(columns).T
    matrix,_=complete_tda_A(static,eri,c,no,exchange_fraction=.486);values,vectors=eigh(matrix)
    q=c[:,:no].T@s.overlap@transition@s.overlap@c[:,no:];q/=np.linalg.norm(q)
    overlaps=abs(q.ravel()@vectors);root=int(np.argmax(overlaps));margin=float(overlaps[root]-np.max(np.delete(overlaps,root),initial=0.))
    return dict(excitation_ev=float(values[root]*EV),root=root+1,overlap=float(overlaps[root]),margin=margin,
        qualified=bool(overlaps[root]>=.7 and margin>=.05),orbitals='own stationary corrected GFN2 ground orbitals',
        operator='complete singlet TDA A including nonlocal exchange correction',lambda_exchange=.9,xc='M06-2X',response_exact_exchange=.486)

def predict_state(case,out,row):
    dest=out/'states'/row['id'];dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json').exists():return
    s=case['s'];no=case['no'];transition=target_transition(case,row);parents=[]
    old_joint=load_relative_pair_roks(case['src']/row['id']/'stationary.npz',request_hash='independent:relative',require_gradient=False)
    for i,profile in enumerate(['F','Q']):
        parent=dest/profile;parent.mkdir(exist_ok=True);current=0.;step=.125;accepted=parent/'accepted.npz'
        oldpath=case['src']/['full_response','qualified_spectral_response'][i]/'anchor/model.npz'
        with np.load(oldpath) as f:data={k:f[k].copy() for k in ['operators','charge','spin','q0','linear']}
        oldref=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,data,sha(oldpath))
        g=ground(case['src']/f'grounds/{i}',case['calc'],case['xyz'],s,oldref)
        earlier=Path('/home/enzo/MOUSE-transferability-20261101/stationary_analytic/orthogonal')/row['cohort']/row['molecule']/row['id']/f'parent_{i}/final.npz'
        with np.load(earlier) as a:c=a['coefficients'].copy()
        if accepted.exists():
            with np.load(accepted) as a:current=float(a['t']);c=a['coefficients'].copy()
            _,g,_=response_and_ground(case,out,profile,current,g)
        trace=json.loads((parent/'continuation.json').read_text()) if (parent/'continuation.json').exists() else []
        attempts=0
        while current<1.-1e-12 and attempts<64:
            attempts+=1;target=min(1.,current+step)
            response,trial_ground,context=response_and_ground(case,out,profile,target,g)
            cg=trial_ground.coefficients[0];q=cg[:,:no].T@s.overlap@transition@s.overlap@cg[:,no:];q/=np.linalg.norm(q)
            trial=parent/f'from_{current:.8f}_to_{target:.8f}'
            result=solve_stationary(context,case['eri'],no,c,cg,q,trial)
            if not result['qualified']:
                trace.append(dict(t_from=current,t_target=target,step=step,status='subdivide',kkt=result['kkt_max'],fidelity=result['fidelity']))
                write(parent/'continuation.json',trace);step*=.5
                if step<1/256:break
                continue
            with np.load(trial/'final.npz') as a:c=a['coefficients'].copy()
            current=target;g=trial_ground
            np.savez_compressed(parent/'accepted.tmp.npz',t=current,coefficients=c)
            (parent/'accepted.tmp.npz').replace(accepted)
            trace.append(dict(t=current,status='accepted',kkt=result['kkt_max'],fidelity=result['fidelity'],gap_ev=(result['energy_hartree']-g.energy_hartree)*EV))
            write(parent/'continuation.json',trace)
            print(row['molecule'],row['id'],profile,'t=',current,flush=True)
        if current<1.-1e-12:
            parents.append(dict(profile=profile,qualified=False,reached_t=current,checkpoint_retained=True));continue
        response,g,context=response_and_ground(case,out,profile,1.,g)
        cg=g.coefficients[0];q=cg[:,:no].T@s.overlap@transition@s.overlap@cg[:,no:];q/=np.linalg.norm(q)
        final=solve_stationary(context,case['eri'],no,c,cg,q,parent/'final')
        final.update(profile=profile,gap_ev=(final['energy_hartree']-g.energy_hartree)*EV,
            tda=tda(context,g,case['eri'],s,no,transition),ground_energy_hartree=g.energy_hartree)
        write(parent/'result.json',final);parents.append(final)
    qualified=all(p['qualified'] for p in parents)
    report=dict(id=row['id'],molecule=row['molecule'],family=row['family'],qualified=qualified,parents=parents,
        energy_labels_used=False,weights=[.5,.5],lambda_exchange=.9,offset_ev=0,host=HOST,threads=2,
        scope='new stationary embedded anchor; separately optimized, ground-orthogonal rank-one ROKS parents',
        source_sha256=sha(__file__),protocol_sha256=sha(DEST/'protocol.json'))
    if qualified:report['excitation_ev']=.5*sum(p['gap_ev'] for p in parents)
    write(dest/'result.json',report)

def predict(cohort,name):
    out=DEST/cohort/name;assert (out/'calibration_complete.json').exists()
    case=prepare(cohort,name)
    case['calc'].evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC forbidden in prepared model prediction'))
    # A prepared prediction must never invoke a full DFT SCF or rebuild its response.
    scf.hf.kernel=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('DFT SCF forbidden in prepared prediction'))
    rows=[dict(row,cohort=cohort) for row in json.loads((OUT/cohort/'manifest.json').read_text())['states'] if row['molecule']==name]
    for row in rows:predict_state(case,out,row)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('cohort');a.add_argument('name');args=a.parse_args()
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():predict(args.cohort,args.name)
