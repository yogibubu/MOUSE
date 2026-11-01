"""Constrained screened interaction control; no numerical energy labels."""
from basis import *
from nosi_spectrum import spectrum
from pauli import normal_ordered_one_body,fock
from screening import screened
from matrix_gfn2.state_interaction import Hamiltonian
ROOT=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-response-consistent-nosi-20261101');BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-frontier-cisd-20261101');DESTINATION=ROOT/'constrained_screening'

def run(cohort,name):
    out=DESTINATION/cohort/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    d=prepare_basis(cohort,name);no=d['no'];c=d['canonical'];s=d['case']['s'];co=c[:,:no];cv=c[:,no:];source=d['source'];holes=[];particles=[]
    for row in d['rows']:
        with np.load(source/'coherent_states'/row['id']/'final/final.npz') as z:cs=z['coefficients'].copy()
        holes.append(co.T@s.overlap@cs[:,0]);particles.append(cv.T@s.overlap@cs[:,no])
    operators=out/'operators.npz'
    if operators.exists():
        with np.load(operators) as z:h=z['h'].copy();w=z['eri'].copy()
        certificate=json.loads((out/'screening.json').read_text())
    else:
        fg,_=d['ctx'].fock_and_energy(np.array([co@co.T]*2));levels=np.diag(c.T@fg[0]@c)
        ys=d['sr']@c;w,certificate=screened(d['eri'],ys,levels,no,holes,particles)
        p=2*(d['sr']@co)@(d['sr']@co).T;target=d['si']@fg[0]@d['si'];h=normal_ordered_one_body(w,target,p)
        assert np.max(abs(fock(w,h,p)-target))<1e-8
        np.savez_compressed(operators,h=h,eri=w);write(out/'screening.json',certificate)
    with np.load(BASE/cohort/name/'checkpoint.npz') as z:diagonal=np.diag(z['matrix']).copy();old_s=z['metric'].copy();four=z['four_spin'].copy()
    old=d['old'];oldreport=json.loads((old/'result.json').read_text());oldpred={r['id']:r for r in oldreport['predictions']}
    with np.load(old/'checkpoint.npz') as z:reference=np.column_stack([z['coefficients'][:,oldpred[row['id']]['root']] for row in d['rows']])
    results=[]
    for label,one in [('normal_ordered',h),('bare_one_body',d['h'])]:
        ham=Hamiltonian(one,w);basis=d['basis'];n=len(basis);metric=np.zeros((n,n));matrix=np.zeros((n,n))
        for i in range(n):
            for j in range(i+1):
                ov,value=ham.csf(basis[i],basis[j]);metric[i,j]=metric[j,i]=ov;matrix[i,j]=matrix[j,i]=value
        assert np.max(abs(metric-old_s))<1e-8
        matrix-=matrix[0,0]*metric;residual=diagonal-np.diag(matrix);matrix+=.5*metric*(residual[:,None]+residual[None,:]);matrix[-2:,-2:]=four
        projected,e,z=spectrum(matrix,metric,reference,d['rows'],True);full,fe,fz=spectrum(matrix,metric,reference,d['rows'],False)
        np.savez_compressed(out/(label+'.npz'),metric=metric,matrix=matrix,energies=e,coefficients=z,full_energies=fe,full_coefficients=fz)
        results.append(dict(one_body=label,primary=projected,full_control=full));print(name,label,projected['predictions'],flush=True)
    write(out/'result.json',dict(results=results,screening=certificate,source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'crpa_protocol.json'),labels=d['labels']))
if __name__=='__main__':
    with threadpool_limits():
        p=json.loads((BASE/'protocol.json').read_text())
        for cohort,name in p['cases']+[['prospective_cisd',r['molecule']] for r in p['prospective_cases']]+[['post_kernel_validation',r['molecule']] for r in json.loads((ROOT/'prospective_protocol.json').read_text())['cases']]:run(cohort,name)
