"""Frozen frontier augmentation, without fitted energies or new DFT SCF."""
from pathlib import Path
import sys
ROOT=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-response-consistent-nosi-20261101')
NATIVE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-native-reference-20261101')
sys.path.insert(0,'/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-cam-response-20261101')
from predict import *
from native_batch import install_native_batch
from nosi_coherent import closed,open_singlet
from matrix_gfn2.state_interaction import Hamiltonian
from matrix_gfn2.state_interaction import four_open_shell_singlets as csfs, four_open_shell_spin_matrix as spin_matrix

def prepare_basis(cohort,name):
    out=ROOT/cohort/name;out.mkdir(parents=True,exist_ok=True)
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
        if sum(i>=no for i in pair)>1:continue
        indices=core+list(pair);b=closed(sr@canonical,indices)
        maximum=max(abs(ham.csf(oldb,b)[0]) for oldb in basis[:base_size])
        additions.append(dict(kind='paired',indices=pair,maximum_retained_overlap=maximum,added=maximum<.7))
        if maximum>=.7:continue
        occupation=np.zeros((2,n));occupation[:,indices]=1
        basis.append(b);diagonal.append(energy(canonical,occupation)-eg);labels.append('pair_'+str(pair))
    for paired in sites:
        for opens in combinations([i for i in sites if i!=paired],2):
            if 2*(paired>=no)+sum(i>=no for i in opens)>2:continue
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
    return dict(case=case,source=source,old=old,ctx=ctx,cg=cg,canonical=canonical,sr=sr,si=si,h=h,eri=eri,basis=basis,labels=labels,rows=rows,no=no)
