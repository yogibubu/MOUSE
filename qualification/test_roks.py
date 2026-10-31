"""Algebraic singlet and derivative tests; no molecular calculations."""
from dataclasses import replace
import numpy as np
import pytest
from matrix_gfn2 import run_roks_singlet, ShellSpinPolarization
from matrix_apoc import DeterminantOrbitals,DeltaSCFState,SpinAdaptedSinglet,singlet_csf_overlap,StateContinuityError,follow_spin_adapted_singlet


def singlet(nclosed=0):
    n = nclosed+2
    ca=np.zeros(n);ca[:nclosed+1]=1
    cb=np.zeros(n);cb[:nclosed]=1;cb[-1]=1
    w = DeterminantOrbitals(np.array([2*nclosed+2]),np.zeros((1,3)),np.eye(n),
        (np.eye(n),np.eye(n)),(ca,cb),np.array([0]),model='test')
    return SpinAdaptedSinglet(1,DeltaSCFState(1,w))


def test_csf_is_annihilated_by_spin_raising_operator_with_closed_pairs():
    # Explicit fermionic second quantization, independent of the declared S^2.
    for nclosed in (0,1,2):
        csf=singlet(nclosed);n=nclosed+2
        raised={}
        for component in csf.components:
            oa,ob=component.wavefunction.mo_occupations
            bit=sum(1<<i for i in np.flatnonzero(oa)) + sum(1<<(n+i) for i in np.flatnonzero(ob))
            for i in range(n):
                if not bit & (1<<(n+i)) or bit & (1<<i):
                    continue
                sign=(-1)**((bit & ((1<<(n+i))-1)).bit_count())
                removed=bit ^ (1<<(n+i))
                sign*=(-1)**((removed & ((1<<i)-1)).bit_count())
                out=removed | (1<<i)
                raised[out]=raised.get(out,0)+sign/np.sqrt(2)
        assert all(abs(x)<1e-14 for x in raised.values())
        assert csf.spin_squared == 0
        assert singlet_csf_overlap(csf,csf,np.eye(n))[0] == pytest.approx(1)


def test_csf_rejects_uhf_or_extra_open_shells_and_tracks_coherent_sum():
    a=singlet();c=a.mixed.wavefunction.mo_coefficients[0].copy()
    c[:,0]*=-1
    w=replace(a.mixed.wavefunction,mo_coefficients=(np.eye(2),c))
    with pytest.raises(ValueError,match='common'):
        SpinAdaptedSinglet(2,DeltaSCFState(2,w))
    with pytest.raises(StateContinuityError):
        follow_spin_adapted_singlet(a,(a,replace(a,state_id=2)),np.eye(2))
    # Interchange the two singly occupied orbitals: the CSF is identical.
    w=replace(a.mixed.wavefunction,mo_occupations=a.mixed.wavefunction.mo_occupations[::-1])
    b=SpinAdaptedSinglet(2,DeltaSCFState(2,w))
    assert singlet_csf_overlap(a,b,np.eye(2))[0] == 1


def test_roks_reoptimizes_common_orbitals_and_uses_multiplet_functional():
    spin=ShellSpinPolarization(np.eye(2),[0,1],[[-.2,0],[0,-.3]])
    h=np.array([[-.7,.08],[.08,-.3]])
    def functional(p):
        v,e,_=spin.evaluate(p)
        return h[None]+v,np.sum(p*h)+e
    result=run_roks_singlet(np.eye(2),np.eye(2),[[1,0],[0,1]],functional)
    em=functional(result.densities())[1];et=functional(result.densities(triplet=True))[1]
    assert result.energy_hartree == pytest.approx(2*em-et)
    assert result.orbital_residual < 1e-9
    assert result.spin_squared == 0


@pytest.mark.parametrize('mode',['deltascf','roks'])
def test_analytic_orbital_response_schur_in_nonorthogonal_metric(mode):
    from matrix_gfn2 import run_deltascf
    from matrix_gfn2.deltascf_analytic_hessian import _response_blocks
    h0=np.array([[-.7,.08],[.08,-.3]])
    h1=np.array([[.11,-.04],[-.04,.07]])
    h2=np.array([[.06,.02],[.02,-.03]])
    s1=np.array([[.1,.2],[.2,-.05]])
    s2=np.array([[.02,.01],[.01,-.03]])
    occ=np.array([[1,0],[0,1]])
    coupling=.09
    def at(r):
        s=np.eye(2)+r*s1+.5*r*r*s2
        spin=ShellSpinPolarization(s,[0,1],[[-.2,0],[0,-.3]])
        h=h0+r*h1+.5*r*r*h2
        def callback(p):
            total=p.sum(axis=0);v,e,_=spin.evaluate(p)
            return h[None]+coupling*total[None]+v, np.sum(total*h)+.5*coupling*np.sum(total*total)+e
        return s,spin,callback
    s,spin,callback=at(0.)
    if mode=='deltascf':
        result=run_deltascf(s,np.array([np.eye(2)]*2),occ,callback,
            density_tolerance=1e-12,orbital_tolerance=1e-12,energy_tolerance=1e-14,maximum_iterations=1000)
        cs=list(result.coefficients);components=[(1.,occ,result.fock)]
    else:
        result=run_roks_singlet(s,np.eye(2),occ,callback)
        cs=[result.common_coefficients]
        components=[(2.,occ,result.fock_mixed),(-1.,result.triplet_occupations,result.fock_triplet)]
    explicit=[]
    for _,o,_ in components:
        allcs=cs*2 if len(cs)==1 else cs
        p=sum((c*oo) @ c.T for c,oo in zip(allcs,o))
        explicit.append((h1[None],np.array([[np.sum(p*h2)]])))
    rr,a,b=_response_blocks(cs,components,s1[None],s2[None,None],explicit,
        [lambda p:coupling*p]*len(components),spin)
    analytic=(rr-b.T @ np.linalg.solve(a,b))[0,0]
    energies=[];step=2e-4
    for r in [-step,0,step]:
        ss,_,fun=at(r);allcs=cs*2 if len(cs)==1 else cs
        projected=[]
        for c in allcs:
            values,vectors=np.linalg.eigh(c.T @ ss @ c)
            projected.append(c @ ((vectors*values**-.5) @ vectors.T))
        if mode=='deltascf':
            trial=run_deltascf(ss,np.array(projected),occ,fun,
                density_tolerance=1e-12,orbital_tolerance=1e-12,energy_tolerance=1e-14,maximum_iterations=1000)
        else:
            trial=run_roks_singlet(ss,projected[0],occ,fun)
        energies.append(trial.energy_hartree)
    numerical=(energies[0]-2*energies[1]+energies[2])/step**2
    assert analytic == pytest.approx(numerical,abs=2e-7)


def test_bs_to_roks_seed_is_invariant_to_occupied_mo_reordering_and_rotations():
    from matrix_gfn2.roks import restricted_singlet_seed
    s=np.eye(4);occ=np.array([[1,1,0,0],[1,0,1,0]])
    c=np.array([np.eye(4)]*2)
    seed=restricted_singlet_seed(s,c,occ)
    rotated=c.copy()
    for spin,indices,angle in [(0,[0,1],.71),(1,[0,2],-.83)]:
        block=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        rotated[spin][:,indices]=rotated[spin][:,indices] @ block
    other=restricted_singlet_seed(s,rotated,occ)
    def descriptor(coefficients):
        w=DeterminantOrbitals(np.array([4]),np.zeros((1,3)),s,(coefficients,coefficients),
            tuple(occ),np.array([0]),model='test')
        return SpinAdaptedSinglet(1,DeltaSCFState(1,w))
    assert singlet_csf_overlap(descriptor(seed),descriptor(other),s)[0] == pytest.approx(1)


def test_spin_pure_result_cannot_be_mutated_or_constructed_with_invalid_orbitals():
    spin=ShellSpinPolarization(np.eye(2),[0,1],[[-.2,0],[0,-.3]])
    h=np.diag([-.7,-.3])
    def callback(p):
        v,e,_=spin.evaluate(p)
        return h[None]+v,np.sum(p*h)+e
    result=run_roks_singlet(np.eye(2),np.eye(2),[[1,0],[0,1]],callback)
    with pytest.raises(ValueError):
        result.common_coefficients[0,0]=2.
    with pytest.raises(ValueError,match='spin-pure'):
        replace(result,common_coefficients=2*np.eye(2))
    np.testing.assert_allclose(result.spin_densities_ao[0],result.spin_densities_ao[1])


def test_local_analytic_orbital_jacobian_matches_independent_rotation_differences():
    from scipy.linalg import expm
    from matrix_gfn2.roks_orbital_response import orbital_residual_jacobian
    n=4;spin=ShellSpinPolarization(np.eye(n),range(n),-.12*np.eye(n))
    h=np.diag([-.8,-.5,-.2,.1]);h[0,2]=h[2,0]=.04
    c=expm(np.array([[0,.1,0,0],[-.1,0,.2,0],[0,-.2,0,.1],[0,0,-.1,0]]))
    left,right=np.triu_indices(n,1)
    for occ in [np.array([[1,1,0,0],[1,0,1,0]]),np.array([[1,1,1,0],[1,0,0,0]])]:
        def residual(coeff):
            p=np.array([(coeff*o)@coeff.T for o in occ]);v,_,_=spin.evaluate(p)
            f=h[None]+.07*p.sum(axis=0)[None]+v
            q=sum((coeff.T@ff@coeff)*o[None] for ff,o in zip(f,occ))
            return 2*(q[left,right]-q[right,left]),f
        r,f=residual(c)
        j=orbital_residual_jacobian(c,occ,f,lambda p:.07*p,spin,left,right)
        finite=[]
        for i,k in zip(left,right):
            g=np.zeros((n,n));g[i,k]=1;g[k,i]=-1
            finite.append((residual(c@expm(1e-5*g))[0]-residual(c@expm(-1e-5*g))[0])/2e-5)
        assert np.max(np.abs(j-np.array(finite).T))<1e-9
