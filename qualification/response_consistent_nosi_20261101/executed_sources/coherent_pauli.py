"""Apply the same spin-preserving Pauli correction to diagonal and SI energies."""
from basis import *
from pauli import project_preserving_spin
from nosi_spectrum import spectrum
from matrix_gfn2.state_interaction import Hamiltonian
ROOT=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-response-consistent-nosi-20261101');DESTINATION=ROOT/'coherent_pauli'

def model(cohort,name,d,out):
    source=d['source'];case=d['case'];s=case['s'];si=d['si']
    path=out/'model.npz'
    if not path.exists():
        with np.load(source/'coherent_common/model.npz') as z:data={k:z[k].copy() for k in ['operators','charge','spin','q0','linear']}
        with np.load(ROOT/'pauli_common'/cohort/name/'operators.npz') as z:charge=z['charge'].copy();spin=z['spin'].copy()
        _,cp,sp,_=project_preserving_spin(charge,spin)
        b=si@(.5*(s.overlap@data['operators']+data['operators']@s.overlap))@si;flat=b.reshape(len(b),-1)
        dual=np.linalg.solve(flat@flat.T,flat);delta=dual@(cp-charge).reshape(len(si)**2,-1)@dual.T
        data['charge']+=.5*(delta+delta.T)
        assert np.max(abs((flat.T@delta@flat).reshape(charge.shape)-(cp-charge)))<1e-8
        np.savez_compressed(path,**data)
    with np.load(path) as z:data={k:z[k].copy() for k in ['operators','charge','spin','q0','linear']}
    r=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,data,sha(path))
    return prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=r)

def run(cohort,name):
    out=DESTINATION/cohort/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'retained_orbital_diagnostic.json').exists():return
    d=prepare_basis(cohort,name);ctx=model(cohort,name,d,out);no=d['no'];n=len(d['cg']);s=d['case']['s'];si=d['si'];sr=d['sr'];basis=d['basis'];labels=d['labels']
    occupations=np.zeros((2,n));occupations[:,:no]=1
    def energy(y,occ):
        c=si@y;return float(ctx.fock_and_energy(np.array([(c*x)@c.T for x in occ]))[1])
    diagonal=[];newbasis=[]
    # Every basis CSF is closed or a two-term open singlet except the final
    # four-spin block. Recover occupied spin densities directly from its terms.
    for b in basis[:-2]:
        if len(b)==1:
            _,(a,bb)=b[0];p=np.array([si@a@a.T@si,si@bb@bb.T@si]);diagonal.append(float(ctx.fock_and_energy(p)[1]))
        else:
            _,(a,bb)=b[0];pa=si@a@a.T@si;pb=si@bb@bb.T@si;em=float(ctx.fock_and_energy(np.array([pa,pb]))[1])
            # The union/intersection of the orthogonal occupied subspaces
            # gives the high-spin partner without assuming canonical labels.
            common=a.T@bb;u,sv,vh=np.linalg.svd(common);closed=a@u[:,sv>1-1e-7];opened=np.column_stack((a@u[:,sv<1-1e-7],bb@vh.T[:,sv<1-1e-7]));q=si@closed@closed.T@si;spin=si@opened@opened.T@si
            et=float(ctx.fock_and_energy(np.array([q+spin,q]))[1]);diagonal.append(2*em-et)
    with np.load(ROOT/'spin_preserving'/cohort/name/'operators.npz') as z:h=z['h'].copy();w=z['eri'].copy()
    # Spin response is unchanged; only the common four-spin charge energy moves.
    with np.load(ROOT/'pauli_common'/cohort/name/'checkpoint.npz') as z:old_four=z['matrix'][-2:,-2:].copy()
    canonical=d['canonical'];balanced=occupations.copy();balanced[:,no-2:no+2]=.5
    old_charge=float(d['ctx'].fock_and_energy(np.array([(canonical*x)@canonical.T for x in balanced]))[1]);new_charge=float(ctx.fock_and_energy(np.array([(canonical*x)@canonical.T for x in balanced]))[1])
    old_eg=float(d['ctx'].fock_and_energy(np.array([d['cg'][:,:no]@d['cg'][:,:no].T]*2))[1]);eg=diagonal[0];four=old_four+(old_eg+new_charge-old_charge)*np.eye(2)
    diagonal.extend(np.diag(four));ham=Hamiltonian(h,w);size=len(basis);metric=np.zeros((size,size));matrix=np.zeros((size,size))
    for i in range(size):
        for j in range(i+1):
            ov,value=ham.csf(basis[i],basis[j]);metric[i,j]=metric[j,i]=ov;matrix[i,j]=matrix[j,i]=value
    residual=np.array(diagonal)-np.diag(matrix);matrix+=.5*metric*(residual[:,None]+residual[None,:]);matrix-=eg*metric;matrix[-2:,-2:]=four-eg*np.eye(2)
    old=d['old'];oldreport=json.loads((old/'result.json').read_text());oldpred={r['id']:r for r in oldreport['predictions']}
    with np.load(old/'checkpoint.npz') as z:reference=np.column_stack([z['coefficients'][:,oldpred[row['id']]['root']] for row in d['rows']])
    full,fe,fc=spectrum(matrix,metric,reference,d['rows'],False);projected,pe,pc=spectrum(matrix,metric,reference,d['rows'],True)
    for row in full['predictions']:
        row['stationary_ground_reference_gap_ev']=row['gap_ev'];row['gap_ev']-=float(fe[0]*EV)
        row['qualified']=bool(row['overlap']>=.7 and row['margin']>=.05 and row['gap_ev']>0)
    write(out/'retained_orbital_diagnostic.json',dict(primary=full,projected_control=projected,stationary_orbitals_of_new_model=False,labels=labels,ground_lowering_ev=float(fe[0]*EV),diagonal_gaps_ev=(np.array(diagonal)-eg)*EV,source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'coherent_pauli_protocol.json')))
    print(name,full['predictions'],flush=True)
if __name__=='__main__':
    with threadpool_limits():
        p=json.loads((Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-frontier-cisd-20261101')/'protocol.json').read_text())
        cases=p['cases']+[['prospective_cisd',r['molecule']] for r in p['prospective_cases']]+[['post_kernel_validation',r['molecule']] for r in json.loads((ROOT/'prospective_protocol.json').read_text())['cases']]
        for cohort,name in cases:run(cohort,name)
