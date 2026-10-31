"""REKS(2,2) fixed-orbital feasibility screen, not a new excitation protocol.

Six determinant energies use the existing spin-polarized functional. Published
occupation interpolation delta=0.4 is fixed, never fitted to excitation energies.
This only measures ground correlation available in a selected active pair;
subtracting it from a separately optimized ROKS energy is NOT a REKS prediction.
Reference: Filatov et al., JCP 147, 034113 (2017), DOI 10.1063/1.4994542;
Lee et al., JCTC 15, 3021 (2019), DOI 10.1021/acs.jctc.9b00132.
"""
from remote_bootstrap import BASE,HOST
from full_response import *
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

DELTA=.4

def interpolation(x):
    y=4*x*(1-x)
    if y<=0:return 0.,float('nan'),float('nan')
    p=(2+DELTA-y)/(2*(1+DELTA));value=y**p
    a=p/y-np.log(y)/(2*(1+DELTA));b=-1/((1+DELTA)*y)-p/y**2
    return value,value*a*(4-8*x),value*((a*a+b)*(4-8*x)**2-8*a)

def occupations(base,hole,particle):
    core=np.asarray(base).copy();core[[hole,particle]]=0
    rows=[]
    for indices in [((hole,),(hole,)),((particle,),(particle,)),
                    ((hole,),(particle,)),((particle,),(hole,)),
                    ((hole,particle),()),((),(hole,particle))]:
        o=np.array([core,core]);o[0,list(indices[0])]=1;o[1,list(indices[1])]=1;rows.append(o)
    return np.asarray(rows)

def coefficients(x):
    g=interpolation(x)[0]
    return np.array([x,1-x,-.5*g,-.5*g,.5*g,.5*g])

def optimize_occupation(e):
    k=.5*(e[2]+e[3]-e[4]-e[5]);gap=e[0]-e[1]
    candidates=[.5,1.]
    if k>0:
        def derivative(x):return gap-k*interpolation(x)[1]
        lo,hi=.5,1-1e-14
        if derivative(lo)*derivative(hi)<0:candidates.append(brentq(derivative,lo,hi,xtol=5e-15))
    values=[float(coefficients(x)@e) for x in candidates];i=int(np.argmin(values))
    return candidates[i],values[i]

def cases():
    result=[c for c in plan() if c['name'] in ['nitroxyl','uracil']]
    for group,names in [('B',['acetaldehyde','acetone','formamide']),('C',['glyoxal','pyrazine','tetrazine']),('D',['acrolein','pyrimidine','nitrosomethane'])]:
        result.extend(json.loads((BASE/('prepared_'+group)/n/'case.json').read_text()) for n in names)
    return result

if __name__=='__main__':
    root=BASE/'results/reks_probe';root.mkdir(exist_ok=True);rows=[]
    write(root/'protocol.json',dict(host=HOST,script_sha256=digest(__file__),delta=DELTA,profiles=['full_response','qualified_spectral_response'],
          active_pair='existing tracked hole and particle',scope='fixed-orbital ground-correlation screen on exposed data; no new excitation predictions',fit=False))
    lib.num_threads(2)
    with threadpool_limits(limits=2,user_api='blas'):
        for case in cases():
            r0,xyz=state_from(case['targets'][0]);s=r0.reference_state
            calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
            ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r0.ground_energy_hartree)
            native=prepare_gfn2_deltascf(calc,xyz,ground_state=ground)
            for profile in ['full_response','qualified_spectral_response']:
                out=BASE/'results'/profile/case['name']
                with np.load(out/'model.npz') as f:data={k:f[k].copy() for k in f.files};request=str(f['request_hash'])
                ref=VerticalReference(s,native.kernel.partition,data,request);ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground,spin_reference=ref)
                with np.load(out/'tda.npz') as f:c=f['coefficients']
                for t in case['targets']:
                    # Canonical corrected-ground MOs need maximum-overlap orbital
                    # labels, rather than silently reusing old column indices.
                    from scipy.optimize import linear_sum_assignment
                    mapping=np.empty(len(c),int)
                    for sector in [np.flatnonzero(s.alpha_occupations==1),np.flatnonzero(s.alpha_occupations==0)]:
                        overlap=abs(s.coefficients[:,sector].T@s.overlap@c[:,sector])**2
                        ii,jj=linear_sum_assignment(-overlap);mapping[sector[ii]]=sector[jj]
                    h,p=mapping[t['hole']],mapping[t['particle']]
                    occ=occupations(s.alpha_occupations,h,p)
                    e=np.array([ctx.fock_and_energy(np.array([(c*o)@c.T for o in oo]))[1] for oo in occ])
                    assert max(abs(e[2]-e[3]),abs(e[4]-e[5]))<1e-9
                    x,energy=optimize_occupation(e)
                    row=dict(molecule=case['name'],state=t['id'],profile=profile,fon=[2*x,2-2*x],
                             ground_lowering_ev=float((e[0]-energy)*EV),microstate_relative_ev=((e-e[0])*EV).tolist(),
                             hole_overlap=float(abs(s.coefficients[:,t['hole']]@s.overlap@c[:,h])),particle_overlap=float(abs(s.coefficients[:,t['particle']]@s.overlap@c[:,p])),
                             occupation_curvature_hartree=None if x==1 else float(-.5*(e[2]+e[3]-e[4]-e[5])*interpolation(x)[2]))
                    rows.append(row);write(root/'summary.json',dict(host=HOST,rows=rows,scope='feasibility only; no orbital reoptimization or excitation validation'))
                    print('REKS PROBE',row,flush=True)
