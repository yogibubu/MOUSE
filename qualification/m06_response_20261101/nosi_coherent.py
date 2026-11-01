"""Oracle spin-adapted nonorthogonal-state-interaction diagnostic."""
from pathlib import Path
import sys
sys.path.insert(0,'/home/enzo/MOUSE-m06-response-20261101')
from predict import *
sys.path.insert(0,'/home/enzo/MOUSE-nosi-20261101')
from slater_condon import csf_element

HERE=Path('/home/enzo/MOUSE-m06-response-20261101/nosi_coherent')

def closed(c,indices):
    occupied=c[:,indices]
    return [(1.,(occupied,occupied))]

def open_singlet(c,no):
    a=c[:,list(range(1,no))+[no]];b=c[:,:no]
    return [(2**-.5,(a,b)),(2**-.5,(b,a))]

def run(cohort,name):
    folder=HERE/cohort/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    case=prepare(cohort,name);s=case['s'];no=case['no'];n=len(s.overlap);source=DEST/cohort/name
    with np.load(source/'coherent_common/model.npz') as z:data={k:z[k].copy() for k in ['operators','linear','charge','spin','q0']}
    response=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,data,sha(source/'coherent_common/model.npz'))
    ctx=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=response)
    ground_path=source/'coherent_common/ground/stationary.npz'
    if not ground_path.exists():ground_path=source/'coherent_continuation_models/t_1.00000000/ground/stationary.npz'
    with np.load(ground_path) as z:cg=z['result_coefficients'][0].copy()
    w,u=eigh(s.overlap);sr=(u*np.sqrt(w))@u.T;si=(u/w**.5)@u.T
    mol=case['mol'];hfull=mol.intor('int1e_kin')+mol.intor('int1e_nuc')
    j,k=scf.hf.get_jk(mol,case['core_density'],hermi=1)
    h=si@case['lift'].T@(hfull+j-.5*k)@case['lift']@si
    eri=np.einsum('pqrs,pi,qj,rk,sl->ijkl',case['eri'],si,si,si,si,optimize=True)
    def energy(c,occ):return float(ctx.fock_and_energy(np.array([(c*x)@c.T for x in occ]))[1])
    ground_occ=np.zeros((2,n));ground_occ[:,:no]=1;eg=energy(cg,ground_occ)
    basis=[closed(sr@cg,list(range(no)))];diagonal=[0.];labels=['S0'];rows=[]
    targets=[row for row in json.loads((OUT/cohort/'manifest.json').read_text())['states'] if row['molecule']==name]
    doubles=[]
    for row in targets:
        state=source/'coherent_states'/row['id'];result=json.loads((state/'result.json').read_text());assert result['qualified']
        path=state/'final/final.npz'
        assert path.exists(),path
        with np.load(path) as z:c=z['coefficients'].copy()
        mixed=ground_occ.copy();mixed[0,0]=0;mixed[0,no]=1
        triplet=np.array([np.maximum(*mixed),np.minimum(*mixed)])
        es=2*energy(c,mixed)-energy(c,triplet)
        assert abs(es-result['energy_hartree'])<1e-9
        basis.append(open_singlet(sr@c,no));diagonal.append(es-eg);labels.append(row['id']);rows.append(row['id'])
        double=ground_occ.copy();double[:,0]=0;double[:,no]=1
        doubles.append((closed(sr@c,list(range(1,no))+[no]),energy(c,double)-eg,row['id']+'_pair_double'))
    for b,e,label in doubles:basis.append(b);diagonal.append(e);labels.append(label)
    size=len(basis);metric=np.zeros((size,size));hf=np.zeros_like(metric)
    for i in range(size):
        for jj in range(i+1):
            ov,value=csf_element(basis[i],basis[jj],h,eri)
            metric[i,jj]=metric[jj,i]=ov;hf[i,jj]=hf[jj,i]=value
    assert np.max(abs(np.diag(metric)-1))<1e-8
    hf-=hf[0,0]*metric;residual=np.asarray(diagonal)-np.diag(hf)
    matrix=hf+.5*metric*(residual[:,None]+residual[None,:])
    mw,mu=eigh(metric);keep=mw>1e-8;orth=mu[:,keep]/np.sqrt(mw[keep])
    e,z=eigh(orth.T@matrix@orth);z=orth@z;overlaps=abs(metric@z)
    assigned,roots=linear_sum_assignment(-overlaps[1:1+len(rows),1:])
    predictions=[]
    for index,root in zip(assigned,roots+1):
        ov=float(overlaps[index+1,root]);margin=ov-float(np.max(np.delete(overlaps[index+1,1:],root-1),initial=0.))
        predictions.append(dict(id=rows[index],root=int(root),gap_ev=float((e[root]-e[0])*EV),overlap=ov,margin=margin,
            qualified=bool(ov>=.7 and margin>=.05 and overlaps[0,0]>=.7)))
    np.savez_compressed(folder/'checkpoint.npz',metric=metric,hf=hf,matrix=matrix,energies=e,coefficients=z,overlaps=overlaps,diagonal=diagonal)
    report=dict(host=HOST,labels=labels,predictions=predictions,ground_character=float(overlaps[0,0]),metric_eigenvalues=mw,
        all_gaps_ev=(e-e[0])*EV,uncoupled_diagonal_ev=np.asarray(diagonal)*EV,
        weights=[.5,.5],lambda_exchange=.9,offset_ev=0,energy_labels_used=False,
        driver_sha256=sha(__file__),slater_condon_sha256=sha(Path('/home/enzo/MOUSE-nosi-20261101/slater_condon.py')),protocol_sha256=sha(DEST/'coherent_reference_protocol.json'),
        scope='new CSF-level overlap-weighted residual functional; unoptimized pair-double configurations; no nuclear derivative claim')
    write(folder/'result.json',report);print(name,predictions,flush=True)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        run('independent','Criegee_Intermediate')
        run('independent_challenge','Furofuran')
