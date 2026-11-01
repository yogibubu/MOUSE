"""Pauli-compatible common off-diagonal response, frozen diagonal functional."""
from nosi_spectrum import spectrum
from basis import *
from pauli import project,normal_ordered_one_body,fock
from matrix_gfn2.state_interaction import Hamiltonian
ROOT=Path('/home/enzo/MOUSE-response-consistent-nosi-20261101');BASE=Path('/home/enzo/MOUSE-frontier-cisd-20261101');DESTINATION=ROOT/'pauli_common'

def response(case,d):
    c=d['canonical'];si=d['si'];no=d['no'];n=len(si);ctx=d['ctx'];basis=[]
    for i in range(n):
        for j in range(i+1):
            x=np.zeros((n,n));x[i,j]=x[j,i]=1. if i==j else 2**-.5;basis.append(x)
    b=np.array(basis);pa=si@b@si;state=SimpleNamespace(**vars(case['s']));state.coefficients=c
    kernel=ctx.density_response_kernel(state);fields=kernel.response_batch(pa) if hasattr(kernel,'response_batch') else np.array([kernel(x) for x in pa])
    charge=np.einsum('aij,bij->ab',b,si@fields@si)
    spin_fields=ctx.polarization.response_batch(np.stack((.5*pa,-.5*pa),axis=1))[:,0]
    spin=np.einsum('aij,bij->ab',b,si@spin_fields@si)
    assert max(np.max(abs(charge-charge.T)),np.max(abs(spin-spin.T)))<1e-8
    charge=(b.reshape(len(b),-1).T@charge@b.reshape(len(b),-1)).reshape((n,)*4)
    spin=(b.reshape(len(b),-1).T@spin@b.reshape(len(b),-1)).reshape((n,)*4)
    w,cp,sp,removed=project(charge,spin)
    y=d['sr']@c;pg=2*y[:,:no]@y[:,:no].T;fg,_=ctx.fock_and_energy(np.array([c[:,:no]@c[:,:no].T]*2));target=si@fg[0]@si;h=normal_ordered_one_body(w,target,pg)
    assert np.max(abs(fock(w,h,pg)-target))<1e-9
    return dict(h=h,eri=w,charge=charge,spin=spin,compatible_charge=cp,compatible_spin=sp,removed=removed,target=target)

def run(cohort,name):
    out=DESTINATION/cohort/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    d=prepare_basis(cohort,name);basis=d['basis'];case=d['case'];operator=out/'operators.npz'
    if operator.exists():
        with np.load(operator) as z:arrays={k:z[k].copy() for k in z.files}
    else:arrays=response(case,d);np.savez_compressed(operator,**arrays)
    ham=Hamiltonian(arrays['h'],arrays['eri']);n=len(basis);s=np.zeros((n,n));h=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            ov,value=ham.csf(basis[i],basis[j]);s[i,j]=s[j,i]=ov;h[i,j]=h[j,i]=value
    with np.load(BASE/cohort/name/'checkpoint.npz') as z:oldh=z['matrix'].copy();olds=z['metric'].copy();four=z['four_spin'].copy()
    assert np.max(abs(s-olds))<1e-8
    h-=h[0,0]*s;diagonal=np.diag(oldh)-np.diag(h);h+=.5*s*(diagonal[:,None]+diagonal[None,:]);h[-2:,-2:]=four
    assert np.max(abs(np.diag(h-oldh)))<1e-8
    old=d['old'];oldreport=json.loads((old/'result.json').read_text());oldpred={r['id']:r for r in oldreport['predictions']}
    with np.load(old/'checkpoint.npz') as z:reference=np.column_stack([z['coefficients'][:,oldpred[row['id']]['root']] for row in d['rows']])
    projected,pe,pc=spectrum(h,s,reference,d['rows'],True);full,fe,fc=spectrum(h,s,reference,d['rows'],False)
    report=dict(primary=projected,full_control=full,labels=d['labels'],relative_response_projection_norm=float(np.linalg.norm(arrays['removed'])/(np.linalg.norm(arrays['charge'])+np.linalg.norm(arrays['spin']))),native_diagonals_preserved=True,lambda_exchange=.9,weights=[.5,.5],energy_offsets_ev=0,source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'pauli_protocol.json'))
    np.savez_compressed(out/'checkpoint.npz',matrix=h,metric=s,energies=pe,coefficients=pc,full_energies=fe,full_coefficients=fc)
    write(out/'result.json',report);print(name,projected['predictions'],flush=True)
if __name__=='__main__':
    with threadpool_limits():
        p=json.loads((BASE/'protocol.json').read_text())
        for cohort,name in p['cases']+[['prospective_cisd',r['molecule']] for r in p['prospective_cases']]:run(cohort,name)
