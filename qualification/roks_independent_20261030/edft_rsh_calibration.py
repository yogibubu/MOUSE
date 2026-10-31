"""Frozen RSH response plus explicit density-driven ensemble coupling.

Research adaptation, NOT the full GX24 functional: retain existing DFT-referenced
first derivative and native GFN2 higher density orders, replace the curvature
by 3/8 short-range HF + full long-range HF (omega=.2), PBE correlation, 5/8
PBE exchange. Lambda=.9 acts once on HF response only. The ensemble Hartree
coupling does not receive this exchange scaling. Published xi=.32 is fixed.
"""
from explicit_coupling_probe import get_eri
from full_response import *
from full_fast import features
from threadpoolctl import threadpool_limits
ROOT=BASE/'results/edft_rsh';ROOT.mkdir(exist_ok=True)
OMEGA=.2;ALPHA=.375;XI=.32;LAMBDA=.9

def calibrate(case,s,native):
    name=case['name'];dest=ROOT/'calibration'/name;dest.mkdir(parents=True,exist_ok=True)
    old=BASE/'results/full_response'/name;source=old/'model.npz'
    request=key(dict(script=digest(__file__),source=digest(source),omega=OMEGA,alpha=ALPHA,lambda_exchange=LAMBDA,xi=XI))
    final=dest/'model.npz'
    if final.exists():
        with np.load(final) as f:assert str(f['request_hash'])==request;return {k:f[k].copy() for k in f.files},request
    start=time.perf_counter()
    with np.load(source) as f:data={k:f[k].copy() for k in f.files}
    operators=data['operators'];n=len(s.overlap);dual=np.array([2*solve_sylvester(s.overlap,s.overlap,a) for a in operators])
    eri,eri_hash=get_eri(case,s)
    if (old/'embedding.npz').exists():
        with np.load(old/'embedding.npz') as f:lift=f['embedding']
    else:
        with np.load(Path('/home/enzo/MOUSE-roks-repair-20261030/results/spin_kernel')/name/'kernel.npz') as f:lift=f['embedding']
    mol=scf.chkfile.load_mol(str(Path(case['reference'])/'backend.chk'));mol.max_memory=40000
    lrfile=dest/'long_range_eri.npz'
    if lrfile.exists():
        with np.load(lrfile) as f:assert str(f['request_hash'])==request;lr=f['eri']
    else:
        with mol.with_range_coulomb(OMEGA):
            lr=ao2mo.incore.general(mol.intor('int2e',aosym='s8'),(lift,lift,lift,lift),compact=False).reshape(n,n,n,n)
        np.savez_compressed(lrfile,request_hash=request,eri=lr)
    hf=dest/'hf_response.npz'
    if hf.exists():
        with np.load(hf) as f:assert str(f['request_hash'])==request;charge=f['charge'];spin=f['spin']
    else:
        # The old HF response is .9*.25; add only the difference.
        delta=LAMBDA*((ALPHA-.25)*eri+(1-ALPHA)*lr)
        d=dual.reshape(len(dual),-1)
        dk=-.5*(d@delta.transpose(0,2,1,3).reshape(n*n,n*n)@d.T)
        charge=data['charge']+dk;spin=data['spin']+dk
        np.savez_compressed(hf,request_hash=request,charge=charge,spin=spin)
    with np.load(Path(case['reference'])/'reference.npz') as f:c=f['coefficients'];occ=f['occupations']
    ncore=int(np.sum(s.atomic_numbers-np.asarray(native.calculator._valence))/2);core=c[:,np.flatnonzero(occ==2)[:ncore]]
    p0=(s.coefficients*(s.alpha_occupations+s.beta_occupations))@s.coefficients.T;dm=core@core.T+.5*lift@p0@lift.T
    mf=dft.UKS(mol);mf.verbose=0;mf.small_rho_cutoff=0;mf.grids.level=3;mf.grids.build(with_non0tab=True)
    progress=dest/'quadrature_progress.npz';completed=0
    if progress.exists():
        with np.load(progress) as f:assert str(f['request_hash'])==request;charge=f['charge'];spin=f['spin'];completed=int(f['completed_blocks'])
    with threadpool_limits(limits=8,user_api='blas'):
        for index,(ao,mask,weight,coords) in enumerate(mf._numint.block_loop(mol,mf.grids,mol.nao_nr(),deriv=1,max_memory=256)):
            if index<completed:continue
            rho=mf._numint.eval_rho(mol,ao,dm,mask,'GGA',hermi=1)
            fxc=mf._numint.eval_xc_eff('GGA_X_PBE',np.array([rho,rho]),deriv=2,xctype='GGA')[2]
            fc=.25*(fxc[0,:,0]+fxc[0,:,1]+fxc[1,:,0]+fxc[1,:,1])
            fs=.25*(fxc[0,:,0]-fxc[0,:,1]-fxc[1,:,0]+fxc[1,:,1]);phi=ao@lift
            for lo in range(0,len(weight),4096):
                hi=min(lo+4096,len(weight));feat=features(phi[:,lo:hi],dual);flat=feat.transpose(1,0,2).reshape(len(operators),-1)
                for target,fx in [(charge,fc),(spin,fs)]:
                    weighted=(np.einsum('ijg,jag->aig',fx[:,:,lo:hi],feat,optimize=True)*weight[None,None,lo:hi]).reshape(len(operators),-1)
                    target+=(.25-ALPHA)*(flat@weighted.T)
            temp=progress.with_suffix('.tmp.npz');np.savez_compressed(temp,request_hash=request,charge=charge,spin=spin,completed_blocks=index+1);temp.replace(progress)
            print('RSH BLOCK',name,index+1,flush=True)
    data.update(charge=.5*(charge+charge.T),spin=.5*(spin+spin.T));data.pop('request_hash',None)
    np.savez_compressed(final,request_hash=request,**data)
    write(dest/'calibration.json',dict(host=HOST,request_hash=request,source_sha256=digest(__file__),old_model_sha256=digest(source),eri_sha256=eri_hash,
          omega=OMEGA,alpha=ALPHA,xi=XI,lambda_exchange=LAMBDA,seconds=time.perf_counter()-start,
          model='GFN2 with RSH curvature and explicit ensemble coupling; not full GX24',first_derivative='existing DFT reference, unchanged',energy_offset=0.,nuclear_derivatives=False))
    return data,request
