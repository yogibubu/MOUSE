"""Oracle-only stationary frozen-core M06-2X reference in the retained embedding."""
from pathlib import Path
import sys
R=Path('/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration/final_validation')
sys.path.insert(0,str(R))
from common import *
import fcntl,argparse,resource
DEST=Path('/home/enzo/MOUSE-m06-response-20261101')

def prepare(cohort,name):
    src=OUT/cohort/'predictions'/name;ref=OUT/cohort/'references'/name
    raw=loadpoint(src/'native_ground.npz');s=canonical_state(raw);calc=calculator(s)
    xyz=s.coordinates_bohr/ANGSTROM_TO_BOHR;ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s)
    with np.load(ref/'reference.npz') as a:cd=a['coefficients'].copy();od=a['occupations'].copy();sd=a['overlap'].copy();ed=a['orbital_energies'].copy()
    with np.load(src/'full_response/anchor/embedding.npz') as a:lift=a['embedding'].copy()
    with np.load(src/'full_response/anchor/projected_eri.npz') as a:eri=a['eri'].copy()
    with np.load(src/'full_response/anchor/model.npz') as a:old={k:a[k].copy() for k in ['operators','linear','charge','spin','q0']}
    mol=scf.chkfile.load_mol(str(ref/'backend.chk'));no=s.electron_count//2;nc=np.count_nonzero(od==2)-no
    core=cd[:,np.flatnonzero(od==2)[:nc]];core_density=2*core@core.T
    assert np.max(abs(lift.T@sd@lift-s.overlap))<1e-8
    assert not nc or np.max(abs(core.T@sd@lift))<1e-8
    return dict(src=src,ref=ref,raw=raw,s=s,calc=calc,xyz=xyz,ctx=ctx,cd=cd,od=od,sd=sd,ed=ed,
                lift=lift,eri=eri,old=old,mol=mol,no=no,core_density=core_density)

class CompactDFT:
    """Exact embedded frozen-core M06-2X functional up to a fixed core constant.

    Reuse projected ERIs and the independently archived native-density Fock.
    No SCF iteration requires a new full-space J/K contraction.
    """
    def __init__(self,case,folder):
        self.case=case;self.folder=folder;folder.mkdir(parents=True,exist_ok=True)
        self.mf=dft.RKS(case['mol']);self.mf.xc='M06-2X';self.mf.verbose=0
        self.mf.small_rho_cutoff=0.;self.mf.max_memory=2000;self.mf.grids.level=3;self.mf.grids.verbose=0
        self.mf.grids.build(with_non0tab=True);self.chunks=[]
        for block,(ao,mask,weight,coords) in enumerate(self.mf._numint.block_loop(case['mol'],self.mf.grids,case['mol'].nao_nr(),deriv=1,max_memory=256)):
            path=folder/f'grid_{block:03d}.npz'
            if path.exists():
                with np.load(path) as a:phi=a['phi'].copy();rho_core=a['rho_core'].copy();saved_weight=a['weight'].copy()
                assert np.array_equal(saved_weight,weight)
            else:
                phi=ao@case['lift'];rho_core=self.mf._numint.eval_rho(case['mol'],ao,case['core_density'],mask,'MGGA',hermi=1,with_lapl=False)
                np.savez_compressed(path,phi=phi,rho_core=rho_core,weight=weight)
            self.chunks.append((phi,rho_core,weight))
        core_j,core_k=scf.hf.get_jk(case['mol'],case['core_density'],hermi=1)
        physical=case['mol'].intor('int1e_kin')+case['mol'].intor('int1e_nuc')+core_j-.27*core_k
        self.hcore=case['lift'].T@physical@case['lift'];self.cache={}

    @staticmethod
    def rho(phi,p):
        rho=np.empty((5,phi.shape[1]));rho[0]=np.einsum('gi,ij,gj->g',phi[0],p,phi[0],optimize=True)
        for k in range(1,4):rho[k]=2*np.einsum('gi,ij,gj->g',phi[k],p,phi[0],optimize=True)
        rho[4]=.5*sum(np.einsum('gi,ij,gj->g',phi[k],p,phi[k],optimize=True) for k in range(1,4))
        return rho

    def xc(self,p):
        energy=0.;v=np.zeros_like(p)
        for phi,core,weight in self.chunks:
            rho=core+self.rho(phi,p);exc,vxc,*_=self.mf._numint.eval_xc_eff('M06-2X',rho,deriv=1,xctype='MGGA')
            energy+=float(weight@(rho[0]*exc));v+=phi[0].T@((weight*vxc[0])[:,None]*phi[0])
            for k in range(1,4):
                part=phi[k].T@((weight*vxc[k])[:,None]*phi[0]);v+=part+part.T
            for k in range(1,4):v+=.5*phi[k].T@((weight*vxc[4])[:,None]*phi[k])
        return energy,.5*(v+v.T)

    def jk(self,p):
        eri=self.case['eri']
        return np.einsum('ijkl,kl->ij',eri,p,optimize=True),np.einsum('ikjl,kl->ij',eri,p,optimize=True)

    def evaluate(self,p):
        if 'p' not in self.cache or not np.array_equal(p,self.cache['p']):
            ex,vxc=self.xc(p);j,k=self.jk(p)
            e=float(np.sum(self.hcore*p)+.5*np.sum(p*j)-.135*np.sum(p*k)+ex)
            self.cache=dict(p=p.copy(),energy=e,fock=self.hcore+j-.27*k+vxc)
        return self.cache['fock'],self.cache['energy']

def anchor(cohort,name):
    out=DEST/cohort/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'anchor_complete.json').exists():return
    start=time.monotonic();case=prepare(cohort,name);functional=CompactDFT(case,out/'grid')
    s=case['s'];no=case['no'];small=scf.hf.RHF(case['mol']);small.verbose=0;small.chkfile=None
    small.get_hcore=lambda *a,**kw:functional.hcore
    small.get_ovlp=lambda *a,**kw:s.overlap
    def veff(mol=None,dm=None,*args,**kwargs):return functional.evaluate(dm)[0]-functional.hcore
    small.get_veff=veff
    small.energy_elec=lambda dm=None,h1e=None,vhf=None:(functional.evaluate(dm)[1],0.)
    small.energy_nuc=lambda *a,**kw:0.
    def occupations(energies,coefficients=None):
        o=np.zeros(len(energies));o[np.argsort(energies)[:no]]=2;return o
    small.get_occ=occupations;small.max_cycle=100;small.conv_tol=1e-11;small.conv_tol_grad=1e-8;small.diis_space=10
    checkpoint=out/'scf_checkpoint.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as a:dm=a['density'].copy()
    else:dm=loadpoint(case['src']/'grounds/0/stationary.npz').density_ao
    history=json.loads((out/'scf_history.json').read_text()) if (out/'scf_history.json').exists() else []
    def callback(env):
        np.savez_compressed(checkpoint,density=env['dm'])
        history.append(dict(cycle=len(history),energy=float(env['e_tot']),gradient_norm=float(env['norm_gorb'])))
        write(out/'scf_history.json',history)
    small.callback=callback;small.kernel(dm0=dm)
    dm=small.make_rdm1();f,e=functional.evaluate(dm);energies,c=eigh(f,s.overlap)
    p=2*c[:,:no]@c[:,:no].T;residual=float(np.max(abs((c.T@f@c)[:no,no:])))
    change=float(np.max(abs(p-dm)));assert small.converged and residual<1e-8 and change<1e-7
    np.savez_compressed(out/'embedded_ground.npz',coefficients=c,orbital_energies=energies,density=dm,fock=f,hcore=functional.hcore,energy_up_to_fixed_core_constant=e)
    # Independent full-AO M06-2X Fock check, evaluated once rather than every SCF cycle.
    full_dm=case['core_density']+case['lift']@dm@case['lift'].T
    vf=functional.mf.get_veff(case['mol'],full_dm)
    check=case['lift'].T@(functional.mf.get_hcore()+vf)@case['lift']
    error=float(np.max(abs(check-f)));assert error<2e-8,(name,error)
    rng=np.random.default_rng(41);direction=rng.normal(size=dm.shape);direction=.5*(direction+direction.T);direction/=np.linalg.norm(direction)
    eps=1e-5;derivative=(functional.evaluate(dm+eps*direction)[1]-functional.evaluate(dm-eps*direction)[1])/(2*eps)
    derivative_error=abs(derivative-np.sum(f*direction));assert derivative_error<2e-7
    write(out/'anchor_complete.json',dict(host=HOST,threads=2,stationary_orbital_fock_error=residual,density_fixed_point_error=change,
        independent_full_ao_fock_error=error,energy_fock_derivative_error=derivative_error,
        wall_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        frozen_core=True,xc='M06-2X',scf_exchange_fraction=.54,source_sha256=sha(__file__),energy_labels_used=False,
        reference_sha256=sha(case['ref']/'reference.npz'),embedding_sha256=sha(case['src']/'full_response/anchor/embedding.npz'),
        protocol_sha256=sha(DEST/'protocol.json'),energy_origin='fixed core constant omitted; no excitation offset'))
    print(name,'stationary embedded DFT verified',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('cohort');a.add_argument('name');args=a.parse_args()
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():anchor(args.cohort,args.name)
