"""Full charge/spin response at the verified stationary embedded density."""
from anchor import *
from matrix_gfn2.roks_orbital_response import orbital_residual_jacobian

def features(phi,dual):
    n=phi.shape[2];g=phi.shape[1];d=dual.reshape(len(dual),n*n)
    base=np.array([(d@(phi[k,:,:,None]*phi[0,:,None,:]).reshape(g,n*n).T)*(1. if k==0 else 2.) for k in range(4)])
    tau=.5*sum(d@(phi[k,:,:,None]*phi[k,:,None,:]).reshape(g,n*n).T for k in range(1,4))
    return np.concatenate([base,tau[None]],axis=0)

def calibrate(cohort,name):
    out=DEST/cohort/name
    if (out/'calibration_complete.json').exists():return
    assert (out/'anchor_complete.json').exists();start=time.monotonic()
    case=prepare(cohort,name);s=case['s'];ctx=case['ctx'];no=case['no']
    with np.load(out/'embedded_ground.npz') as a:c=a['coefficients'].copy();p=a['density'].copy();levels=a['orbital_energies'].copy();target=a['fock'].copy()
    functional=CompactDFT(case,out/'grid');a=case['old']['operators'];b=.5*(s.overlap@a+a@s.overlap)
    dual=np.array([2*solve_sylvester(s.overlap,s.overlap,x) for x in a]);nf=len(a);eri=case['eri'];n=len(s.overlap)
    packed=dual.reshape(nf,n*n)
    j=packed@eri.reshape(n*n,n*n)@packed.T;k=packed@eri.transpose(0,2,1,3).reshape(n*n,n*n)@packed.T
    full_charge=j-.243*k;full_spin=-.243*k;completed=0
    progress=out/'response_progress.npz'
    if progress.exists():
        with np.load(progress) as f:full_charge=f['charge'].copy();full_spin=f['spin'].copy();completed=int(f['completed'])
    for block,(phi,core,weight) in enumerate(functional.chunks):
        if block<completed:continue
        rho=.5*(core+functional.rho(phi,p))
        fxc=functional.mf._numint.eval_xc_eff('M06-2X',np.array([rho,rho]),deriv=2,xctype='MGGA')[2]
        fc=.25*(fxc[0,:,0]+fxc[0,:,1]+fxc[1,:,0]+fxc[1,:,1]);fs=.25*(fxc[0,:,0]-fxc[0,:,1]-fxc[1,:,0]+fxc[1,:,1])
        for lo in range(0,len(weight),4096):
            hi=min(lo+4096,len(weight));feat=features(phi[:,lo:hi],dual);flat=feat.transpose(1,0,2).reshape(nf,-1)
            for result,fx in [(full_charge,fc),(full_spin,fs)]:
                weighted=(np.einsum('ijg,jag->aig',fx[:,:,lo:hi],feat,optimize=True)*weight[None,None,lo:hi]).reshape(nf,-1)
                result+=flat@weighted.T
        temp=progress.with_suffix('.tmp.npz');np.savez_compressed(temp,charge=full_charge,spin=full_spin,completed=block+1);temp.replace(progress)
        print(name,'stationary response block',block+1,'/',len(functional.chunks),flush=True)
    assert max(np.max(abs(full_charge-full_charge.T)),np.max(abs(full_spin-full_spin.T)))<1e-7
    state=SimpleNamespace(**vars(s));state.coefficients=c
    native_kernel=ctx.density_response_kernel(state)
    native_response=native_kernel.response_batch(dual) if hasattr(native_kernel,'response_batch') else np.asarray([native_kernel(x) for x in dual])
    native=np.einsum('aij,bij->ab',dual,native_response)
    charge=full_charge-native;charge=.5*(charge+charge.T);spin=.5*(full_spin+full_spin.T)
    native_f,native_e=ctx.fock_and_energy(np.array([.5*p,.5*p]));q0=np.einsum('aij,ij->a',b,p)
    common=dict(operators=a,charge=charge,spin=spin,q0=q0)
    targets=[target]
    # Q retains the qualified overlap/margin rules, now in stationary canonical orbitals.
    cross=c.T@case['lift'].T@case['sd']@case['cd'];nc=np.count_nonzero(case['od']==2)-no
    selected=np.zeros(n,int);score=np.zeros(n);margin=np.zeros(n)
    for ids,refs in [(np.arange(no),np.flatnonzero(case['od']==2)[nc:]),(np.arange(no,n),np.flatnonzero(case['od']==0))]:
        overlaps=cross[np.ix_(ids,refs)];_,assignment=linear_sum_assignment(-overlaps**2)
        for i,idx in enumerate(ids):
            selected[idx]=refs[assignment[i]];score[idx]=abs(overlaps[i,assignment[i]])
            margin[idx]=score[idx]-np.max(np.delete(abs(overlaps[i]),assignment[i]),initial=0.)
    good=(score>=.7)&(margin>=.05);delta=np.zeros(n)
    source_origin=np.mean(levels[:no]);reference_origin=np.mean(case['ed'][selected[:no]])
    delta[good]=(case['ed'][selected[good]]-reference_origin)-(levels[good]-source_origin)
    shifted=levels+delta;assert np.max(shifted[:no])<np.min(shifted[no:]),'Q transfer loses occupied/virtual gap'
    qtarget=(s.overlap@c*shifted)@c.T@s.overlap;targets.append(qtarget)
    write(out/'Q_mapping.json',dict(indices=selected,overlaps=score,margins=margin,qualified=good,delta_hartree=delta,
        source_centroid=source_origin,reference_centroid=reference_origin,lambda_exchange=.9,gap_scale=1.,offset_ev=0))
    certificates=[]
    for i,(profile,target) in enumerate(zip(['F','Q'],targets)):
        folder=out/profile;folder.mkdir(exist_ok=True);model=folder/'model.npz'
        if not model.exists():np.savez_compressed(model,**common,linear=np.einsum('aij,ij->a',dual,target-native_f[0]))
        with np.load(model) as f:data={key:f[key].copy() for key in ['operators','charge','spin','q0','linear']}
        response=FrozenNonlocalResponseReference.from_vertical_model(s,ctx.kernel.partition,data,sha(model))
        context=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=response)
        ff,ee=context.fock_and_energy(np.array([.5*p,.5*p]));error=float(np.max(abs(ff-target)))
        assert error<2e-8,(profile,error)
        seed=replace(case['raw'],coefficients=np.array([c,c]),fock=ff,energy_hartree=ee,
                     spin_reference_fingerprint=response.fingerprint,reference_state=s,converged=True)
        g=ground(folder/'ground',case['calc'],case['xyz'],s,response,seed=seed)
        drift=float(np.max(abs(g.density_ao-p)));assert drift<2e-6,(profile,drift)
        rng=np.random.default_rng(43);dp=rng.normal(size=(2,n,n));dp=.5*(dp+dp.transpose(0,2,1));dp/=np.linalg.norm(dp)
        kernel=context.density_response_kernel(state)
        linear_response=kernel(dp.sum(0))+context.polarization.response_batch(dp[None])[0]
        eps=1e-5;plus=context.fock_and_energy(np.array([.5*p,.5*p])+eps*dp)[0];minus=context.fock_and_energy(np.array([.5*p,.5*p])-eps*dp)[0]
        derivative_error=float(np.max(abs((plus-minus)/(2*eps)-linear_response)));assert derivative_error<2e-7
        row=dict(profile=profile,fock_error=error,ground_density_drift=drift,
            minimum_ground_curvature=g.minimum_orbital_curvature_hartree,ground_residual=g.orbital_residual,
            response_directional_error=derivative_error,model_sha256=sha(model),response_fingerprint=response.fingerprint)
        write(folder/'calibration_certificate.json',row);certificates.append(row)
    write(out/'calibration_complete.json',dict(host=HOST,threads=2,profiles=certificates,
        response_center='stationary embedded M06-2X density',lambda_exchange=.9,weights=[.5,.5],offset_ev=0,
        wall_seconds=time.monotonic()-start,source_sha256=sha(__file__),anchor_sha256=sha(out/'embedded_ground.npz'),
        protocol_sha256=sha(DEST/'protocol.json'),energy_labels_used=False))
    print(name,'stationary F/Q calibration certified',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('cohort');a.add_argument('name');args=a.parse_args()
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():calibrate(args.cohort,args.name)
