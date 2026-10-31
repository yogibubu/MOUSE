"""Matched remote timing after calibration and S0; exact archived TDA check.

TDA gives all roots for BOTH parent kernels. ROKS gives one coupled state.
Three short repetitions are timing measurements, not production restarts.
"""
from vertical_relative import *
from matrix_gfn2.relative_pair_roks import load_relative_pair_roks
from scipy.linalg import eigh


def benchmark(name):
    folder = OUT/name/'n_pi'
    dest = folder/'matched_timing.json'
    if dest.exists():
        return
    central = load_relative_pair_roks(folder/'stationary.npz',
        request_hash='relative_vertical_control_v1', require_gradient=False)
    responses, contexts, matrices, states = [], [], [], []
    xyz = central.components[0].reference_state.coordinates_bohr/ANGSTROM_TO_BOHR
    atoms = [{1:'H',6:'C',7:'N',8:'O'}[int(z)]
             for z in central.components[0].reference_state.atomic_numbers]
    calc = GFN2Calculator(atoms,temperature_kelvin=0.)
    calc.evaluate = lambda *a,**kw: (_ for _ in ()).throw(AssertionError('SCC invoked'))
    for parent, profile in zip(central.components,['full_response','qualified_spectral_response']):
        path = OLD/'results'/profile/name
        native = prepare_gfn2_deltascf(calc,xyz,orbital_reference=parent.reference_state)
        with np.load(path/'model.npz') as data:
            model = {k:data[k].copy() for k in ['operators','charge','spin','q0','linear']}
        ref = FrozenNonlocalResponseReference.from_vertical_model(parent.reference_state,
            native.kernel.partition,model,hashlib.sha256((path/'model.npz').read_bytes()).hexdigest())
        ctx = prepare_gfn2_deltascf(calc,xyz,orbital_reference=parent.reference_state,spin_reference=ref)
        with np.load(path/'tda.npz') as data:
            c = data['coefficients'].copy()
            archived = data['matrix'].copy()
        occupations = parent.reference_state.alpha_occupations
        dm = np.array([(c*occupations)@c.T]*2)
        fock,_ = ctx.fock_and_energy(dm)
        energies = np.diag(c.T@fock[0]@c)
        state = replace(parent.reference_state, coefficients=c, fock=fock[0],
                        orbital_energies_hartree=energies)
        states.append(state);matrices.append(archived);contexts.append(ctx);responses.append(ref)
    def tda():
        differences=[]
        for state,ctx,archived in zip(states,contexts,matrices):
            c=state.coefficients;occ=np.flatnonzero(state.alpha_occupations>.5)
            virt=np.flatnonzero(state.alpha_occupations<.5)
            ii=np.repeat(occ,len(virt));aa=np.tile(virt,len(occ));ci=c[:,ii];ca=c[:,aa]
            kernel=ctx.density_response_kernel(state);columns=[]
            for i,a in zip(ii,aa):
                p=np.outer(c[:,i],c[:,a]);f=kernel(.5*(p+p.T))
                columns.append(np.einsum('mi,mn,ni->i',ci,f,ca))
            matrix=np.diag(state.orbital_energies_hartree[aa]-state.orbital_energies_hartree[ii])+2*np.array(columns).T
            differences.append(float(np.max(abs(matrix-archived))))
            np.linalg.eigh(matrix)
        if max(differences)>1e-8:
            raise ValueError('TDA kernel/orbitals do not match archived comparison: '+str(differences))
        return differences
    rows=[]
    for repeat in range(3):
        start=time.perf_counter();cpu=time.process_time();errors=tda()
        tda_row=dict(seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu)
        start=time.perf_counter();cpu=time.process_time()
        point=evaluate_relative_pair_roks(calc,xyz,responses,reference_result=central,
            minimum_state_overlap=.7,gradient=False)
        roks_row=dict(seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                      energy_change_hartree=abs(point.energy_hartree-central.energy_hartree))
        rows.append(dict(repeat=repeat,tda_both_all_roots=tda_row,
                         roks_one_state_warm=roks_row,maximum_tda_matrix_error=max(errors)))
    result=dict(host=socket.getfqdn(),molecule=name,threads=2,rows=rows,
        scope='cached calibration, ground states and prepared native kernels; no DFT orbitals; no gradients',
        cold_roks_seconds=json.loads((folder/'result.json').read_text())['seconds'],
        cold_timing_scope='earlier same-host run from separately optimized assigned parents; not simultaneous controlled timing',
        calibration_and_S0_cost_excluded=True)
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        for name in ['nitroxyl','acetaldehyde','acetone','formamide','glyoxal']:
            benchmark(name)
