"""Frozen frontier augmentation, without fitted energies or new DFT SCF."""
from pathlib import Path
import sys
ROOT=Path('/home/enzo/MOUSE-frontier-complete-20261101')
NATIVE=Path('/home/enzo/MOUSE-native-reference-20261101')
sys.path.insert(0,'/home/enzo/MOUSE-cam-response-20261101')
from predict import *
from native_batch import install_native_batch
from nosi_coherent import closed,open_singlet
from fast_slater import Hamiltonian
from spin_four import csfs,matrix as spin_matrix

def run(cohort,name):
    out=ROOT/cohort/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    source=NATIVE/cohort/name;old=NATIVE/'nosi_coherent'/cohort/name
    if not (old/'result.json').exists():print('Awaiting',name,flush=True);return
    case=prepare(cohort,name);s=case['s'];no=case['no'];n=len(s.overlap)
    install_native_batch(case,source)
    with np.load(source/'coherent_common/model.npz') as z:data={k:z[k].copy() for k in ['operators','charge','spin','q0','linear']}
    response=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,data,sha(source/'coherent_common/model.npz'))
    ctx=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=response)
    with np.load(source/'coherent_common/ground/stationary.npz') as z:cg=z['result_coefficients'][0].copy()
    w,u=eigh(s.overlap);sr=(u*np.sqrt(w))@u.T;si=(u/np.sqrt(w))@u.T
    mol=case['mol'];hfull=mol.intor('int1e_kin')+mol.intor('int1e_nuc');j,k=scf.hf.get_jk(mol,case['core_density'],hermi=1)
    h=si@case['lift'].T@(hfull+j-.5*k)@case['lift']@si
    eri=np.einsum('pqrs,pi,qj,rk,sl->ijkl',case['eri'],si,si,si,si,optimize=True)
    ham=Hamiltonian(h,eri)
    def energy(c,occ):return float(ctx.fock_and_energy(np.array([(c*x)@c.T for x in occ]))[1])
    occ=np.zeros((2,n));occ[:,:no]=1;eg=energy(cg,occ)
    basis=[closed(sr@cg,list(range(no)))];diagonal=[0.];labels=['S0'];doubles=[]
    rows=[r for r in json.loads((OUT/cohort/'manifest.json').read_text())['states'] if r['molecule']==name]
    for row in rows:
        with np.load(source/'coherent_states'/row['id']/'final/final.npz') as z:c=z['coefficients'].copy()
        mix=occ.copy();mix[0,0]=0;mix[0,no]=1;trip=np.array([np.maximum(*mix),np.minimum(*mix)])
        basis.append(open_singlet(sr@c,no));diagonal.append(2*energy(c,mix)-energy(c,trip)-eg);labels.append(row['id'])
        double=occ.copy();double[:,0]=0;double[:,no]=1
        doubles.append((closed(sr@c,list(range(1,no))+[no]),energy(c,double)-eg,row['id']+'_pair_double'))
    for b,e,label in doubles:basis.append(b);diagonal.append(e);labels.append(label)
    base_size=len(basis)
    fg,_=ctx.fock_and_energy(np.array([cg[:,:no]@cg[:,:no].T]*2));_,canonical=eigh(fg[0],s.overlap)
    from itertools import combinations
    additions=[];core=list(range(no-2));sites=list(range(no-2,no+2))
    for pair in combinations(sites,2):
        indices=core+list(pair);b=closed(sr@canonical,indices)
        maximum=max(abs(ham.csf(oldb,b)[0]) for oldb in basis[:base_size])
        additions.append(dict(kind='paired',indices=pair,maximum_retained_overlap=maximum,added=maximum<.7))
        if maximum>=.7:continue
        occupation=np.zeros((2,n));occupation[:,indices]=1
        basis.append(b);diagonal.append(energy(canonical,occupation)-eg);labels.append('pair_'+str(pair))
    for paired in sites:
        for opens in combinations([i for i in sites if i!=paired],2):
            indices=core+[paired];first=canonical[:,indices+[opens[0]]];second=canonical[:,indices+[opens[1]]]
            b=[(2**-.5,(sr@first,sr@second)),(2**-.5,(sr@second,sr@first))]
            maximum=max(abs(ham.csf(oldb,b)[0]) for oldb in basis[:base_size])
            additions.append(dict(kind='open',paired=paired,opens=opens,maximum_retained_overlap=maximum,added=maximum<.7))
            if maximum>=.7:continue
            mixed=np.zeros((2,n));mixed[:,indices]=1;mixed[0,opens[0]]=1;mixed[1,opens[1]]=1
            triplet=np.zeros((2,n));triplet[:,indices]=1;triplet[0,list(opens)]=1
            basis.append(b);diagonal.append(2*energy(canonical,mixed)-energy(canonical,triplet)-eg);labels.append(f'open_{paired}_{opens}')
    four_start=len(basis);basis.extend(csfs(sr@canonical,no));labels+=['four_spin_0','four_spin_1']
    balanced=occ.copy();balanced[:,no-2:no+2]=.5
    p0=np.array([(canonical*x)@canonical.T for x in balanced]);ec=float(ctx.fock_and_energy(p0)[1])
    # Spin functional is 1/2 (q_alpha-q_beta)^T K_spin (q_alpha-q_beta).
    operators=.5*(s.overlap@data['operators']+data['operators']@s.overlap)
    sites=canonical[:,no-2:no+2];features=np.einsum('aij,ik,jk->ka',operators,sites,sites)
    pair=features@data['spin']@features.T;four=spin_matrix(ec-eg,pair)
    diagonal.extend(np.diag(four))
    size=len(basis);metric=np.zeros((size,size));hf=np.zeros((size,size))
    for i in range(size):
        for jj in range(i+1):
            ov,value=ham.csf(basis[i],basis[jj]);metric[i,jj]=metric[jj,i]=ov;hf[i,jj]=hf[jj,i]=value
    assert np.max(abs(np.diag(metric)-1))<1e-8
    hf-=hf[0,0]*metric;residual=np.asarray(diagonal)-np.diag(hf)
    matrix=hf+.5*metric*(residual[:,None]+residual[None,:]);matrix[four_start:,four_start:]=four
    with np.load(old/'checkpoint.npz') as z:oldmetric=z['metric'].copy();oldmatrix=z['matrix'].copy();oldc=z['coefficients'].copy()
    errors=dict(metric=float(np.max(abs(metric[:base_size,:base_size]-oldmetric))),matrix=float(np.max(abs(matrix[:base_size,:base_size]-oldmatrix))))
    assert max(errors.values())<1e-8,errors
    mw,mu=eigh(metric);keep=mw>1e-8;orth=mu[:,keep]/np.sqrt(mw[keep]);e,z=eigh(orth.T@matrix@orth);z=orth@z
    ground_weights=abs(metric[0]@z);ground_root=int(np.argmax(ground_weights));assert ground_root==0
    oldreport=json.loads((old/'result.json').read_text());oldrows={r['id']:r for r in oldreport['predictions']}
    oldvectors=np.column_stack([oldc[:,oldrows[r['id']]['root']] for r in rows])
    weights=abs(oldvectors.T@metric[:base_size]@z);assigned,roots=linear_sum_assignment(-weights[:,1:]);predictions=[]
    for i,root in zip(assigned,roots+1):
        ov=float(weights[i,root]);margin=ov-float(np.max(np.delete(weights[i,1:],root-1),initial=0.))
        predictions.append(dict(id=rows[i]['id'],gap_ev=float(e[root]*EV),conventional_si_gap_ev=float((e[root]-e[0])*EV),root=int(root),overlap=ov,margin=margin,qualified=bool(ov>=.7 and margin>=.05)))
    np.savez_compressed(out/'checkpoint.npz',metric=metric,matrix=matrix,hf=hf,energies=e,coefficients=z,diagonal=diagonal,four_spin=four,spin_pair=pair)
    report=dict(predictions=predictions,labels=labels,metric_eigenvalues=mw,auxiliary_si_ground_lowering_ev=float(e[0]*EV),ground_character=float(ground_weights[0]),retained_block_errors=errors,additions=additions,all_eigenvalues_ev=e*EV,source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'protocol.json'),host=HOST,energy_labels_used=False,offset_ev=0)
    write(out/'result.json',report);print(name,predictions,flush=True)

if __name__=='__main__':
    with threadpool_limits():
        for cohort,name in json.loads((ROOT/'protocol.json').read_text())['cases']:run(cohort,name)
