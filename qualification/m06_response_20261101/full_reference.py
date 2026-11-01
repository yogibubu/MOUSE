"""Oracle-only full-basis M06-2X reference and coherent spectral transfer."""
from anchor import *
from scipy.optimize import linear_sum_assignment

def full_reference(cohort,name):
    out=DEST/cohort/name;folder=out/'M06_full_reference';folder.mkdir(exist_ok=True)
    case=prepare(cohort,name);path=folder/'reference.npz';s=case['s'];no=case['no']
    if not path.exists():
        start=time.monotonic();mf=dft.RKS(case['mol']);mf.xc='M06-2X';mf.grids.level=3;mf.small_rho_cutoff=0.;mf.verbose=0
        mf.chkfile=str(folder/'backend.chk');mf.max_cycle=100;mf.conv_tol=1e-10;mf.conv_tol_grad=1e-7
        checkpoint=folder/'backend.chk'
        if checkpoint.exists():
            try:
                old=scf.chkfile.load(str(checkpoint),'scf');seed=(old['mo_coeff']*old['mo_occ'])@old['mo_coeff'].T
            except Exception:seed=(case['cd']*case['od'])@case['cd'].T
        else:seed=(case['cd']*case['od'])@case['cd'].T
        history=json.loads((folder/'history.json').read_text()) if (folder/'history.json').exists() else []
        def callback(env):
            history.append(dict(cycle=len(history),energy=float(env['e_tot']),gradient=float(env['norm_gorb'])))
            write(folder/'history.json',history)
        mf.callback=callback;mf.kernel(dm0=seed);assert mf.converged
        p=mf.make_rdm1();f=mf.get_fock(dm=p);orth=mf.mo_coeff.T@f@mf.mo_coeff;occupied=mf.mo_occ>0
        residual=float(np.max(abs(orth[np.ix_(occupied,~occupied)])));assert residual<1e-6
        np.savez_compressed(path,coefficients=mf.mo_coeff,occupations=mf.mo_occ,orbital_energies=mf.mo_energy,overlap=mf.get_ovlp(),density=p,fock=f)
        write(folder/'certificate.json',dict(host=HOST,xc='M06-2X',converged=mf.converged,occupied_virtual_fock_residual=residual,wall_seconds=time.monotonic()-start,
            source_sha256=sha(__file__),protocol_sha256=sha(DEST/'coherent_reference_protocol.json'),seed_sha256=sha(case['ref']/'reference.npz'),energy_labels_used=False))
    common=out/'coherent_common';common.mkdir(exist_ok=True)
    if (common/'model.npz').exists():return
    with np.load(path) as z:cd=z['coefficients'].copy();od=z['occupations'].copy();ed=z['orbital_energies'].copy()
    with np.load(out/'embedded_ground.npz') as z:c=z['coefficients'].copy();levels=z['orbital_energies'].copy();p=z['density'].copy()
    with np.load(out/'F/model.npz') as z:d={key:z[key].copy() for key in ['operators','charge','spin','q0','linear']}
    cross=c.T@case['lift'].T@case['sd']@cd;nc=np.count_nonzero(od==2)-no;n=len(c)
    selected=np.zeros(n,int);score=np.zeros(n);margin=np.zeros(n)
    for ids,refs in [(np.arange(no),np.flatnonzero(od==2)[nc:]),(np.arange(no,n),np.flatnonzero(od==0))]:
        overlaps=cross[np.ix_(ids,refs)];_,assignment=linear_sum_assignment(-overlaps**2)
        for i,idx in enumerate(ids):
            selected[idx]=refs[assignment[i]];score[idx]=abs(overlaps[i,assignment[i]])
            margin[idx]=score[idx]-np.max(np.delete(abs(overlaps[i]),assignment[i]),initial=0.)
    good=(score>=.7)&(margin>=.05);delta=np.zeros(n)
    delta[good]=(ed[selected[good]]-np.mean(ed[selected[:no]]))-(levels[good]-np.mean(levels[:no]))
    shifted=levels+delta;assert np.max(shifted[:no])<np.min(shifted[no:])
    qtarget=(s.overlap@c*shifted)@c.T@s.overlap
    native_f,_=case['ctx'].fock_and_energy(np.array([.5*p,.5*p]));dual=np.array([2*solve_sylvester(s.overlap,s.overlap,x) for x in d['operators']])
    qlinear=np.einsum('aij,ij->a',dual,qtarget-native_f[0])
    qfolder=out/'Q_M06';qfolder.mkdir(exist_ok=True);np.savez_compressed(qfolder/'model.npz',**dict(d,linear=qlinear))
    np.savez_compressed(common/'model.npz',**dict(d,linear=.5*(d['linear']+qlinear)))
    write(common/'mapping.json',dict(indices=selected,overlaps=score,margins=margin,qualified=good,delta_hartree=delta,
        reference_sha256=sha(path),source_sha256=sha(__file__),protocol_sha256=sha(DEST/'coherent_reference_protocol.json'),weights=[.5,.5],lambda_exchange=.9,offset_ev=0))
    print(name,'coherent M06 reference and Q model retained',flush=True)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        full_reference('independent','Criegee_Intermediate')
        full_reference('independent_challenge','Furofuran')
