"""Oracle-only common spin-pure CAS diagnostic; no reference energies read."""
from pathlib import Path
import sys
sys.path.insert(0, '/home/enzo/MOUSE-stationary-anchor-20261101')
from predict import *
from response_projection import *
from pyscf import fci
from scipy.linalg import null_space
from scipy.optimize import linear_sum_assignment

DESTINATION = Path('/home/enzo/MOUSE-hamiltonian-projection-20261101')

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

def run(cohort, name):
    folder = DESTINATION/cohort/name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'result.json').exists(): return
    case=prepare(cohort,name); s=case['s']; no=case['no']; n=len(s.overlap)
    source=DEST/cohort/name
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
    with np.load(source/'common_functional/ground/stationary.npz') as z:
        print('ground keys',z.files,flush=True)
        target=si@z['result_fock'][0]@si
    v=hamiltonian_projection(charge,spin); h=normal_ordered_one_body(v,target,p)
    assert np.max(abs(fock(v,h,p)-target))<1e-10
    assert np.max(abs(v-v.transpose(2,3,0,1)))<1e-9
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
    identity=np.eye(dim); heff=fci.direct_spin1.absorb_h1e(hc,vc,norb,ne,.5)
    matrix=np.column_stack([fci.direct_spin1.contract_2e(heff,identity[:,i].reshape(nd,nd),norb,ne).ravel() for i in range(dim)])
    ss=np.column_stack([fci.spin_op.contract_ss(identity[:,i].reshape(nd,nd),norb,ne).ravel() for i in range(dim)])
    assert np.max(abs(matrix-matrix.T))<1e-9
    assert np.max(abs(ss@matrix-matrix@ss))<1e-9
    spins,projector=eigh(ss); singlet=projector[:,abs(spins)<1e-8]
    energies,ci=eigh(singlet.T@matrix@singlet); ci=singlet@ci
    spectra=[];characters=[]
    for k in range(1,len(energies)):
        t=fci.direct_spin1.trans_rdm1(ci[:,0].reshape(nd,nd),ci[:,k].reshape(nd,nd),norb,ne)
        t=.5*(t+t.T)
        projected=ct[:,:no].T@active@t@active.T@ct[:,no:]
        norm=np.linalg.norm(projected); normalized=projected/max(norm,1e-20)
        characters.append([abs(float(np.sum(q*normalized))) for q in qs])
        spectra.append(dict(root=k,gap_ev=float((energies[k]-energies[0])*EV),transition_norm=float(norm),spin_squared=float(ci[:,k]@ss@ci[:,k])))
    characters=np.array(characters).T
    indices,roots=linear_sum_assignment(-characters)
    predictions=[]
    for index,root in zip(indices,roots):
        overlap=float(characters[index,root]);margin=overlap-float(np.max(np.delete(characters[index],root)))
        predictions.append(dict(id=rows[index]['id'],**spectra[root],overlap=overlap,margin=margin,qualified=bool(overlap>=.7 and margin>=.05)))
    rc,rs=response_residuals(v,charge,spin)
    report=dict(host=HOST,molecule=name,model='closest representable fermionic response; CASCI(6,6)',orbital_optimization=False,nuclear_derivatives=False,
        energy_labels_used=False,offset_ev=0,weights=[.5,.5],lambda_exchange=.9,projection_charge_residual=float(np.linalg.norm(rc)),projection_spin_residual=float(np.linalg.norm(rs)),
        reference_fock_error=float(np.max(abs(fock(v,h,p)-target))),singlet_dimension=singlet.shape[1],predictions=predictions,spectrum=spectra,source_sha256=sha(__file__),protocol_sha256=sha(DESTINATION/'protocol.json'))
    np.savez_compressed(folder/'checkpoint.npz',h=h,v=v,active=active,core=core,energies=energies+ec,ci=ci,characters=characters)
    write(folder/'result.json',report);print(name,predictions,flush=True)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        run('independent','Criegee_Intermediate')
        run('independent_challenge','Furofuran')
