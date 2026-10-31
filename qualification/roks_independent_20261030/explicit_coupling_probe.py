"""Screen an EDFT-inspired spin functional on existing fixed orbitals.

E_S = E_T + 2 lambda K_ab, lambda=0.9 fixed by the existing protocol.
This is a screened-coupling research extension of the existing hybrid-derived
semiempirical functional, not a claim to implement the full hybrid EDFT theory.
No spin-purification term is added on top, preventing double counting.
The unscaled 2K value is archived as a literature-limit diagnostic only.
"""
from reks_probe import *
from pyscf import ao2mo,scf
ROOT=BASE/'results/explicit_coupling';ROOT.mkdir(exist_ok=True)

def get_eri(case,s):
    old=BASE/'results/full_response'/case['name'];dest=ROOT/case['name'];dest.mkdir(exist_ok=True)
    for p in [old/'projected_eri.npz',old/'partial_quadrature.npz',dest/'projected_eri.npz']:
        if p.exists():
            with np.load(p) as f:
                if 'eri' in f.files:return f['eri'].copy(),digest(p)
    with np.load(Path('/home/enzo/MOUSE-roks-repair-20261030/results/spin_kernel')/case['name']/'kernel.npz') as f:lift=f['embedding']
    mol=scf.chkfile.load_mol(str(Path(case['reference'])/'backend.chk'));n=len(s.overlap)
    eri=ao2mo.incore.general(mol.intor('int2e',aosym='s8'),(lift,lift,lift,lift),compact=False).reshape(n,n,n,n)
    np.savez_compressed(dest/'projected_eri.npz',eri=eri)
    return eri,digest(dest/'projected_eri.npz')

if __name__=='__main__':
    protocol=dict(host=HOST,functional='triplet same-functional energy plus 2 lambda K_ab',lambda_exchange=.9,energy_offsets=False,
                  ground='same corrected closed-shell functional as existing profile',profiles=['full_response','qualified_spectral_response'],
                  status='fixed-orbital exposed diagnostic; not relaxed predictions',references=['10.1063/5.0274509','arXiv:2606.13654'],script_sha256=digest(__file__))
    write(ROOT/'protocol.json',protocol);rows=[];lib.num_threads(2)
    with threadpool_limits(limits=2,user_api='blas'):
        for case in cases():
            r0,xyz=state_from(case['targets'][0]);s=r0.reference_state;eri,eri_hash=get_eri(case,s)
            calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
            ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r0.ground_energy_hartree)
            native=prepare_gfn2_deltascf(calc,xyz,ground_state=ground)
            for profile in protocol['profiles']:
                out=BASE/'results'/profile/case['name']
                with np.load(out/'model.npz') as f:data={k:f[k].copy() for k in f.files};request=str(f['request_hash'])
                ref=VerticalReference(s,native.kernel.partition,data,request);ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground,spin_reference=ref)
                for t in case['targets']:
                    old=json.loads((out/t['id']/'result.json').read_text());r=load_derivative_point(old['checkpoint'],request_hash=old['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
                    a,b=r.common_coefficients[:,t['hole']],r.common_coefficients[:,t['particle']]
                    transition=np.outer(a,b);kab=float(np.einsum('ij,ijkl,kl',transition,eri,transition,optimize=True))
                    em=ctx.fock_and_energy(r.densities())[1];et=ctx.fock_and_energy(r.densities(triplet=True))[1]
                    value=(et+1.8*kab-r.ground_energy_hartree)*EV
                    row=dict(molecule=case['name'],state=t['id'],profile=profile,old_ev=r.excitation_energy_hartree*EV,
                             coupling_hartree=kab,old_singlet_triplet_ev=2*(em-et)*EV,explicit_singlet_triplet_ev=1.8*kab*EV,
                             screened_fixed_orbital_ev=value,unscaled_limit_fixed_orbital_ev=(et+2*kab-r.ground_energy_hartree)*EV,
                             tbe_ev=t['tbe_ev'],eri_sha256=eri_hash,relaxed=False)
                    rows.append(row);print('COUPLING',row,flush=True)
            write(ROOT/'probe.json',dict(host=HOST,rows=rows,scope='nonstationary exposed diagnostic; no fitted offsets or parameters'))
