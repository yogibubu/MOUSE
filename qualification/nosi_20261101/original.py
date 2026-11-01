"""Same state interaction on retained original unmodified F and Q parents."""
from diagnostic import *

def spectrum(metric,matrix,rows,project=False):
    p=np.eye(len(metric))
    if project:
        p=p[:,1:];p[0,:]-=metric[0,1:]/metric[0,0]
    w,u=eigh(p.T@metric@p);keep=w>1e-8;orth=p@u[:,keep]/np.sqrt(w[keep])
    e,z=eigh(orth.T@matrix@orth);z=orth@z;overlaps=abs(metric@z)
    start=0 if project else 1
    assigned,roots=linear_sum_assignment(-overlaps[1:3,start:]);predictions=[]
    for index,root in zip(assigned,roots+start):
        ov=float(overlaps[index+1,root]);margin=ov-float(np.max(np.delete(overlaps[index+1,start:],root-start),initial=0.))
        predictions.append(dict(id=rows[index],root=int(root),gap_ev=float((e[root]-(0 if project else e[0]))*EV),overlap=ov,margin=margin,
            qualified=bool(ov>=.7 and margin>=.05 and (project or overlaps[0,0]>=.7))))
    return dict(predictions=predictions,all_gaps_ev=(e-(0 if project else e[0]))*EV,overlaps=overlaps,ground_character=None if project else float(overlaps[0,0]),metric_eigenvalues=w),e,z

def run_original(cohort,name):
    folder=HERE/'original'/cohort/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    case=prepare(cohort,name);s=case['s'];no=case['no'];n=len(s.overlap)
    w,u=eigh(s.overlap);sr=(u*np.sqrt(w))@u.T;si=(u/w**.5)@u.T
    mol=case['mol'];hfull=mol.intor('int1e_kin')+mol.intor('int1e_nuc');j,k=scf.hf.get_jk(mol,case['core_density'],hermi=1)
    h=si@case['lift'].T@(hfull+j-.5*k)@case['lift']@si
    eri=np.einsum('pqrs,pi,qj,rk,sl->ijkl',case['eri'],si,si,si,si,optimize=True)
    targets=[row for row in json.loads((OUT/cohort/'manifest.json').read_text())['states'] if row['molecule']==name]
    parents=[]
    for parent,profile in enumerate(['full_response','qualified_spectral_response']):
        destination=folder/str(parent);destination.mkdir(exist_ok=True)
        if (destination/'result.json').exists():parents.append(json.loads((destination/'result.json').read_text()));continue
        model=case['src']/profile/'anchor/model.npz'
        with np.load(model) as z:data={k:z[k].copy() for k in ['operators','charge','spin','q0','linear']}
        r=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,data,sha(model))
        ctx=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=r)
        g=ground(case['src']/f'grounds/{parent}',case['calc'],case['xyz'],s,r);cg=g.coefficients[0]
        def energy(c,occ):return float(ctx.fock_and_energy(np.array([(c*x)@c.T for x in occ]))[1])
        occ=np.zeros((2,n));occ[:,:no]=1;eg=energy(cg,occ)
        basis=[closed(sr@cg,list(range(no)))];diagonal=[0.];labels=['S0'];doubles=[]
        for row in targets:
            joint=load_relative_pair_roks(case['src']/row['id']/'stationary.npz',request_hash='independent:relative',require_gradient=False)
            component=joint.components[parent];c=component.common_coefficients
            mix=occ.copy();mix[0,0]=0;mix[0,no]=1;trip=np.array([np.maximum(*mix),np.minimum(*mix)])
            es=2*energy(c,mix)-energy(c,trip)
            assert abs(es-component.energy_hartree)<1e-8
            basis.append(open_singlet(sr@c,no));diagonal.append(es-eg);labels.append(row['id'])
            double=occ.copy();double[:,0]=0;double[:,no]=1
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
        report,e,z=spectrum(metric,matrix,labels[1:3]);projected,pe,pz=spectrum(metric,matrix,labels[1:3],True)
        np.savez_compressed(destination/'checkpoint.npz',metric=metric,hf=hf,matrix=matrix,energies=e,coefficients=z,diagonal=diagonal,projected_energies=pe,projected_coefficients=pz)
        report.update(projected_diagnostic=projected,labels=labels,uncoupled_diagonal_ev=np.asarray(diagonal)*EV,model_sha256=sha(model))
        write(destination/'result.json',report);parents.append(report);print(name,parent,report['predictions'],flush=True)
    predictions=[]
    for i,row in enumerate(targets):
        values=[p['predictions'][i] for p in parents];assert all(v['id']==row['id'] for v in values)
        predictions.append(dict(id=row['id'],gap_ev=.5*sum(v['gap_ev'] for v in values),qualified=all(v['qualified'] for v in values)))
    write(folder/'result.json',dict(predictions=predictions,parents=parents,host=HOST,weights=[.5,.5],lambda_exchange=.9,offset_ev=0,energy_labels_used=False,
        protocol_sha256=sha(HERE/'original_protocol.json'),source_sha256=sha(__file__)))

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        run_original('independent','Criegee_Intermediate')
        run_original('independent_challenge','Furofuran')
