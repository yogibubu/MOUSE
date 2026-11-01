"""Oracle-only common spin-pure CAS diagnostic; no reference energies read."""
from pathlib import Path
import sys
sys.path.insert(0, '/home/enzo/MOUSE-stationary-anchor-20261101')
from predict import *
from pauli import *
from pyscf import fci
from pyscf.fci import direct_nosym
from scipy.linalg import null_space
from scipy.optimize import linear_sum_assignment

DESTINATION = Path('/home/enzo/MOUSE-pauli-projection-20261101')

def sector_basis(vectors, dimension, count, order):
    basis = []
    for vector in list(vectors) + [np.eye(dimension)[:, i] for i in order]:
        v = vector.copy()
        for _ in range(2):
            for b in basis: v -= b * (b @ v)
        norm = np.linalg.norm(v)
        if norm > 1e-8: basis.append(v/norm)
        if len(basis) == count: break
    assert len(basis) == count
    return np.column_stack(basis)

def run(model, cohort, name):
    folder = DESTINATION/model/cohort/name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'result.json').exists(): return
    case=prepare(cohort,name); s=case['s']; no=case['no']; n=len(s.overlap)
    source=Path('/home/enzo/MOUSE-stationary-anchor-20261101' if model=='PBE0' else '/home/enzo/MOUSE-m06-response-20261101')/cohort/name
    with np.load(source/'embedded_ground.npz') as z:
        print('anchor keys',z.files,flush=True)
        c=z['coefficients'].copy()
    eigen, u=eigh(s.overlap); sr=(u*np.sqrt(eigen))@u.T; si=(u/eigen**.5)@u.T
    ct=sr@c; p=2*ct[:,:no]@ct[:,:no].T
    with np.load(source/'F/model.npz') as z: operators=z['operators'].copy()
    b=si@(.5*(s.overlap@operators+operators@s.overlap))@si
    with np.load(source/'response_progress.npz') as z:
        print('response keys',z.files,flush=True)
        charge=np.einsum('aij,ab,bkl->ijkl',b,z['charge'],b,optimize=True)
        spin=np.einsum('aij,ab,bkl->ijkl',b,z['spin'],b,optimize=True)
    model_path=source/('common_functional' if model=='PBE0' else 'coherent_common')/'model.npz'
    with np.load(model_path) as z:data={key:z[key].copy() for key in ['operators','charge','spin','q0','linear']}
    b_ao=.5*(s.overlap@data['operators']+data['operators']@s.overlap)
    p_ao=c[:,:no]@c[:,:no].T*2
    native_f,_=case['ctx'].fock_and_energy(np.array([.5*p_ao,.5*p_ao]))
    field=data['linear']+data['charge']@(np.einsum('aij,ij->a',b_ao,p_ao)-data['q0'])
    target=si@(native_f[0]+np.einsum('a,aij->ij',field,b_ao))@si
    v,cp,sp,incompatible=project(charge,spin);h=normal_ordered_one_body(v,target,p)
    assert np.max(abs(fock(v,h,p)-target))<1e-10
    assert np.max(abs(v-v.transpose(2,3,0,1)))<1e-9
    assert np.max(abs(pair_symmetric(v)-.5*exchange(v)-cp))<1e-9
    assert np.max(abs(-.5*exchange(v)-sp))<1e-9
    rows=[dict(row,cohort=cohort) for row in json.loads((OUT/cohort/'manifest.json').read_text())['states'] if row['molecule']==name]
    qs=[];holes=[];particles=[]
    for row in rows:
        t=target_transition(case,row); q=c[:,:no].T@s.overlap@t@s.overlap@c[:,no:]; q/=np.linalg.norm(q)
        left,_,right=np.linalg.svd(q,full_matrices=False)
        qs.append(q);holes.append(left[:,0]);particles.append(right[0])
    holes=sector_basis(holes,no,3,range(no-1,-1,-1));particles=sector_basis(particles,n-no,3,range(n-no))
    active=np.column_stack([ct[:,:no]@holes,ct[:,no:]@particles])
    core=ct[:,:no]@null_space(holes.T);pc=2*core@core.T
    hc=active.T@fock(v,h,pc)@active
    vc=np.einsum('ijkl,ia,jb,kc,ld->abcd',v,active,active,active,active,optimize=True)
    ec=.5*np.sum(pc*(h+fock(v,h,pc)))
    norb=6;ne=(3,3);nd=20;dim=nd*nd
    identity=np.eye(dim); heff=fci.direct_nosym.absorb_h1e(hc,vc,norb,ne,.5)
    matrix=np.column_stack([fci.direct_nosym.contract_2e(heff,identity[:,i].reshape(nd,nd),norb,ne).ravel() for i in range(dim)])
    ss=np.column_stack([fci.spin_op.contract_ss(identity[:,i].reshape(nd,nd),norb,ne).ravel() for i in range(dim)])
    assert np.max(abs(matrix-matrix.T))<1e-9
    assert np.max(abs(ss@matrix-matrix@ss))<1e-9
    spins,projector=eigh(ss); singlet=projector[:,abs(spins)<1e-8]
    energies,ci=eigh(singlet.T@matrix@singlet); ci=singlet@ci
    spectra=[];characters=[];amplitudes=[]
    for k in range(1,len(energies)):
        t=fci.direct_spin1.trans_rdm1(ci[:,0].reshape(nd,nd),ci[:,k].reshape(nd,nd),norb,ne)
        t=.5*(t+t.T)
        projected=ct[:,:no].T@active@t@active.T@ct[:,no:]
        norm=np.linalg.norm(projected); normalized=projected/max(norm,1e-20)
        characters.append([abs(float(np.sum(q*normalized))) for q in qs]);amplitudes.append([abs(float(np.sum(q*projected))) for q in qs])
        spectra.append(dict(root=k,gap_ev=float((energies[k]-energies[0])*EV),transition_norm=float(norm),spin_squared=float(ci[:,k]@ss@ci[:,k])))
    characters=np.array(characters).T
    amplitudes=np.array(amplitudes).T
    indices,roots=linear_sum_assignment(-amplitudes)
    predictions=[]
    for index,root in zip(indices,roots):
        overlap=float(characters[index,root]);winning=float(amplitudes[index,root]);margin=(winning-float(np.max(np.delete(amplitudes[index],root))))/max(winning,1e-20)
        predictions.append(dict(id=rows[index]['id'],**spectra[root],overlap=overlap,margin=margin,qualified=bool(overlap>=.7 and margin>=.05)))
    rc=cp-charge;rs=sp-spin
    report=dict(host=HOST,molecule=name,model=model+' nearest Pauli-compatible general-nonlocal response; CASCI(6,6)',orbital_optimization=False,nuclear_derivatives=False,
        energy_labels_used=False,offset_ev=0,weights=[.5,.5],lambda_exchange=.9,projection_charge_residual=float(np.linalg.norm(rc)),projection_spin_residual=float(np.linalg.norm(rs)),
        reference_fock_error=float(np.max(abs(fock(v,h,p)-target))),singlet_dimension=singlet.shape[1],predictions=predictions,spectrum=spectra,source_sha256=sha(__file__),protocol_sha256=sha(DESTINATION/'protocol.json'))
    np.savez_compressed(folder/'checkpoint.npz',h=h,v=v,active=active,core=core,energies=energies+ec,ci=ci,characters=characters,amplitudes=amplitudes)
    write(folder/'result.json',report);print(name,predictions,flush=True)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        for model in ['PBE0','M06-2X']:
            run(model,'independent','Criegee_Intermediate')
            run(model,'independent_challenge','Furofuran')
