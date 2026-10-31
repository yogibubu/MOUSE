"""Ground-response expansion diagnostic, with exact E/F/K at retained ground."""
from pair_diagnostic import *

def run(cohort,name):
    src=R/cohort/'predictions'/name
    raw=loadpoint(src/'native_ground.npz');s=canonical_state(raw)
    calc=calculator(s);xyz=s.coordinates_bohr/ANGSTROM_TO_BOHR
    ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s)
    contexts=[];grounds=[];kernels=[]
    for i,profile in enumerate(['full_response','qualified_spectral_response']):
        p=src/profile/'anchor/model.npz'
        with np.load(p) as a:d={k:a[k].copy() for k in ['operators','charge','spin','linear','q0']}
        response=FrozenNonlocalResponseReference.from_vertical_model(s,ctx.kernel.partition,d,sha(p))
        assert (src/f'grounds/{i}/certificate.json').exists()
        g=ground(src/f'grounds/{i}',calc,xyz,s,response)
        context=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=response)
        state=SimpleNamespace(**vars(s));state.coefficients=g.coefficients[0]
        state.alpha_occupations,state.beta_occupations=g.occupations
        contexts.append(context);grounds.append(g);kernels.append(context.density_response_kernel(state))
    calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Unexpected SCC'))
    rows=[r for r in json.loads((R/cohort/'manifest.json').read_text())['states'] if r['molecule']==name]
    for row in rows:
        dest=DEST/'nonlinearity'/cohort/name/row['id']/'result.json'
        if dest.exists():continue
        if not (src/row['id']/'stationary.npz').exists():continue
        projection=json.loads((src/row['id']/'projection.json').read_text())
        if projection['rank']!=1:continue
        joint=load_relative_pair_roks(src/row['id']/'stationary.npz',request_hash='independent:relative',require_gradient=False)
        parents=[]
        for context,kernel,g,component in zip(contexts,kernels,grounds,joint.components):
            p0=g.spin_densities_ao;f0,e0=context.fock_and_energy(p0);c=component.common_coefficients
            energies=[];exact=[];norms=[]
            for occ in [component.occupations,component.triplet_occupations]:
                p=np.array([(c*x)@c.T for x in occ]);dp=p-p0
                response=kernel(dp.sum(0))+context.polarization.evaluate(dp)[0]
                energies.append(float(e0+np.sum(f0*dp)+.5*np.sum(dp*response)))
                exact.append(float(context.fock_and_energy(p)[1]));norms.append(float(np.linalg.norm(dp)))
            no=s.electron_count//2;ov=g.coefficients[0][:,:no].T@s.overlap@c
            dets=[np.linalg.det(ov[:,np.flatnonzero(o)]) for o in component.occupations]
            parents.append(dict(ground_singlet_overlap_squared=float(2*np.prod(dets)**2),exact_gap_ev=(2*exact[0]-exact[1]-e0)*EV,
                quadratic_gap_ev=(2*energies[0]-energies[1]-e0)*EV,
                mixed_remainder_ev=(exact[0]-energies[0])*EV,triplet_remainder_ev=(exact[1]-energies[1])*EV,density_change_norms=norms))
        write(dest,dict(id=row['id'],molecule=name,parents=parents,energy_labels_used=False,
            source_sha256=sha(__file__),scope='frozen orbital decomposition, not an optimized replacement surface'))
        print(name,row['id'],'response decomposition complete',flush=True)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        for cohort,name in [('independent','Criegee_Intermediate'),('independent','Benzoxadiazole'),('independent','Hexatriene'),('independent_challenge','Furofuran')]:
            try:run(cohort,name)
            except Exception:write(DEST/'nonlinearity'/cohort/name/'error.json',dict(traceback=traceback.format_exc()))
