"""Independent finite-difference check of the full M06-2X spin kernel.

Uses exact quadrature Focks at finite rotated spin densities, not the fitted
quadratic response energy. Exchange lambda is accounted for explicitly.
"""
from anchor import *
from scipy.linalg import expm

def validate(cohort,name):
    out=DEST/cohort/name;output=out/'independent_mgga_kernel_validation.json'
    if output.exists():return
    case=prepare(cohort,name);functional=CompactDFT(case,out/'grid');s=case['s'];no=case['no']
    with np.load(out/'embedded_ground.npz') as z:c=z['coefficients'].copy()
    with np.load(out/'F/model.npz') as z:a=z['operators'].copy()
    b=.5*(s.overlap@a+a@s.overlap)
    with np.load(out/'response_progress.npz') as z:charge=z['charge'].copy();spin=z['spin'].copy()
    def exact_fock(dm):
        total=dm.sum(0);j,_=functional.jk(total);f=[]
        for sigma in range(2):
            _,k=functional.jk(dm[sigma]);f.append(functional.hcore+j-.54*k)
        f=np.array(f)
        for phi,core,w in functional.chunks:
            rho=np.array([.5*core+functional.rho(phi,p) for p in dm])
            exc,v= functional.mf._numint.eval_xc_eff('M06-2X',rho,deriv=1,xctype='MGGA')[:2]
            for sigma in range(2):
                f[sigma]+=phi[0].T@((w*v[sigma,0])[:,None]*phi[0])
                for k in range(1,4):
                    part=phi[k].T@((w*v[sigma,k])[:,None]*phi[0]);f[sigma]+=part+part.T
                    f[sigma]+=.5*phi[k].T@((w*v[sigma,4])[:,None]*phi[k])
        return f
    rng=np.random.default_rng(31);direction=rng.normal(size=(len(c),len(c)));direction-=direction.T;direction/=np.linalg.norm(direction)
    pc=c[:,:no]@c[:,:no].T;dc=c@direction;dp=dc[:,:no]@c[:,:no].T+c[:,:no]@dc[:,:no].T
    rows=[]
    for mode,sign in [('charge',1),('spin',-1)]:
        delta=np.array([dp,sign*dp]);dq=np.einsum('aij,sij->sa',b,delta)
        vcharge=charge@dq.sum(0);vspin=spin@(dq[0]-dq[1]);expected=np.einsum('sa,aij->sij',np.array([vcharge+vspin,vcharge-vspin]),b)
        for eps in [2e-5,1e-5]:
            densities=[]
            for pm in [-1,1]:
                cs=[c@expm(pm*eps*k*direction) for k in [1,sign]]
                densities.append(np.array([x[:,:no]@x[:,:no].T for x in cs]))
            fd=(exact_fock(densities[1])-exact_fock(densities[0]))/(2*eps)
            fd+=.054*np.array([functional.jk(x)[1] for x in delta])
            error=float(np.max(abs(fd-expected)));assert error<1e-6,(name,mode,eps,error)
            rows.append(dict(mode=mode,epsilon=eps,max_error_hartree=error))
    write(output,dict(host=HOST,xc='M06-2X',tests=rows,independent='exact unrestricted quadrature Focks at rotated positive spin densities; lambda-exchange correction explicit',source_sha256=sha(__file__)))
    print(name,rows,flush=True)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        validate('independent','Criegee_Intermediate')
        validate('independent_challenge','Furofuran')
