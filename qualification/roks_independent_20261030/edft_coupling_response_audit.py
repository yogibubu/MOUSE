"""Independent scalar derivative and exact HF spin-coupling checks on Oracle."""
from edft_rsh_response import *
from pathlib import Path
case=next(c for c in cases() if c['name']=='nitroxyl');target=case['targets'][0]
r0,xyz=state_from(target);s=r0.reference_state
row=json.loads((BASE/'results/qualified_spectral_response/nitroxyl/n_pi/result.json').read_text())
r=load_derivative_point(row['checkpoint'],request_hash=row['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
c=r.common_coefficients;eri,_=get_eri(case,s);n=len(c);left,right=np.triu_indices(n,1)
lib.num_threads(2)
with threadpool_limits(limits=2,user_api='blas'):
    e,g,h=coupling_response(c,eri,target['hole'],target['particle'],left,right)
    def scalar(coeff):
        a,b=coeff[:,target['hole']],coeff[:,target['particle']]
        return float(np.einsum('i,j,ijkl,k,l',a,b,eri,a,b,optimize=True))
    assert abs(scalar(c)-e)<1e-12
    rng=np.random.default_rng(4289);checks=[]
    for direction in range(4):
        d=rng.normal(size=len(g));d/=np.linalg.norm(d);k=np.zeros((n,n));k[left,right]=d;k[right,left]=-d
        for step in [2e-4,1e-4]:
            ep=scalar(c@expm(step*k));em=scalar(c@expm(-step*k))
            checks.append(dict(direction=direction,step=step,gradient_error=abs((ep-em)/(2*step)-g@d),hessian_error=abs((ep-2*e+em)/step**2-d@h@d)))
    def hf(occ):
        ps=np.array([(c*o)@c.T for o in occ]);p=ps.sum(axis=0)
        coulomb=.5*np.einsum('ij,ijkl,kl',p,eri,p,optimize=True)
        exchange=-.5*sum(np.einsum('ij,ikjl,kl',x,eri,x,optimize=True) for x in ps)
        return float(coulomb+exchange)
    # For common orbitals the two-determinant singlet energy is exactly
    # E_T + 2 K_ab in the unscaled, no-correlation Hartree-Fock limit.
    hf_error=abs(2*(hf(r.occupations)-hf(r.triplet_occupations))-2*e)
    passed=bool(hf_error<1e-10 and all(d['gradient_error']<1e-7 and d['hessian_error']<2e-6 for d in checks))
    report=dict(host=HOST,checks=checks,hf_spin_coupling_identity_error_hartree=hf_error,passed=passed,
                scope='quartic orbital coupling only; analytic electronic Hessian, no nuclear derivatives')
    write(BASE/'results/edft_rsh_consistent/coupling_independent_audit.json',report);print(report,flush=True);assert passed
