"""Check the RI-preconditioned vertical adapter against the full established functional."""
from prepare_fresh import *
from pigment_compact_reference import Compact
from pigment_ri_xc import prepare as auxiliary_response
from matrix_gfn2.nosi_surface import PreparedNOSIModel
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2._quadrature_xc import FrozenQuadratureXC
from matrix_gfn2._nosi_surface_response import MatrixResponse
from matrix_gfn2._roks_matrix_free import solve as solve_roks
from scipy.linalg import expm
from scipy.linalg import solve_sylvester
from dataclasses import replace
from types import SimpleNamespace
from pyscf import df

def run():
 name='Ethylene';cohort='closing_blind';case=prepare(cohort,name);s=case['s'];n=len(s.overlap);no=s.electron_count//2
 root=HERE/'exact_quadrature_validation';root.mkdir(exist_ok=True);grid=root/'grid';grid.mkdir(exist_ok=True)
 old=PreparedNOSIModel(HERE/'disentangled_bath_spectral_block_reference_eta_0.600000'/cohort/name/'model.json')
 i,j=np.tril_indices(n);w=np.where(i==j,1.,2**.5);nf=len(i);E=np.zeros((nf,n,n));E[np.arange(nf),i,j]=1/w;E[np.arange(nf),j,i]=1/w
 with np.load(CAM/cohort/name/'embedded_ground.npz') as z:p=z['density'].copy()
 mf=dft.RKS(case['mol']);mf.xc='CAM-B3LYP';mf.verbose=0;mf.grids.level=3;mf.small_rho_cutoff=0.;mf.grids.build(with_non0tab=True)
 full=df.DF(case['mol']);full.auxbasis='def2-universal-jkfit';full.build();factors=[]
 for block in full.loop(blksize=16):factors.append(ao2mo._ao2mo.nr_e2(block,np.asfortranarray(case['lift']),(0,n,0,n),aosym='s2',mosym='s1').reshape(len(block),n,n))
 paths=[];chunks=[];exact=[np.zeros((nf,nf)),np.zeros((nf,nf))]
 for k,(ao,mask,weight,coordinates) in enumerate(mf._numint.block_loop(case['mol'],mf.grids,case['mol'].nao_nr(),deriv=1,max_memory=128)):
  phi=ao@case['lift'];core=mf._numint.eval_rho(case['mol'],ao,case['core_density'],mask,'GGA',hermi=1);rho=.5*(core+Compact.rho(phi,p));fxc=mf._numint.eval_xc_eff('CAM-B3LYP',np.array([rho,rho]),deriv=2,xctype='GGA')[2]
  fc=.25*(fxc[0,:,0]+fxc[0,:,1]+fxc[1,:,0]+fxc[1,:,1]);fs=.25*(fxc[0,:,0]-fxc[0,:,1]-fxc[1,:,0]+fxc[1,:,1]);path=grid/f'chunk_{k:04d}.npz';np.savez(path,phi=phi,core=core,weight=weight,coordinates=coordinates,charge=fc,spin=fs);paths.append(path);chunks.append(dict(file=str(path.relative_to(root)),sha256=sha(path)))
  for lo in range(0,len(weight),512):
   hi=min(lo+512,len(weight));ph=phi[:,lo:hi];f=np.empty((4,nf,hi-lo));f[0]=w[:,None]*ph[0,:,i]*ph[0,:,j]
   for a in range(1,4):f[a]=w[:,None]*(ph[a,:,i]*ph[0,:,j]+ph[0,:,i]*ph[a,:,j])
   flat=f.transpose(1,0,2).reshape(nf,-1)
   for channel,fx in enumerate([fc[:,:,lo:hi],fs[:,:,lo:hi]]):exact[channel]+=flat@(np.einsum('ijg,jag->aig',fx,f,optimize=True)*weight[None,None,lo:hi]).reshape(nf,-1).T
 small=SimpleNamespace(name=name,n=n,mol=case['mol'],lift=case['lift'],mf=mf,a=np.concatenate(factors),grid=paths,rho=Compact.rho)
 fitted,certificate=auxiliary_response(small,p,i,j,w,root,require_accuracy=False)
 operators=np.array([2*solve_sylvester(s.overlap,s.overlap,e) for e in E]);T=np.linalg.lstsq(old.response.operators.reshape(nf,-1).T,operators.reshape(nf,-1).T,rcond=None)[0].T;inverse=np.linalg.inv(T);r=old.response
 reference=replace(r,operators=operators,pair_charge_hartree=inverse.T@r.pair_charge_hartree@inverse+fitted[0]-exact[0],pair_spin_hartree=inverse.T@r.pair_spin_hartree@inverse+fitted[1]-exact[1],pair_reference_moments=T@r.pair_reference_moments,pair_linear_hartree=inverse.T@r.pair_linear_hartree)
 for channel,key in enumerate(['charge','spin']):np.save(root/(key+'.npy'),fitted[channel])
 write(root/'quadrature.json',dict(coordinates_bohr=s.coordinates_bohr,chunks=chunks));base=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=reference);ctx=FrozenQuadratureXC(base,root/'quadrature.json',root/'charge.npy',root/'spin.npy',reference);original=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=r)
 rng=np.random.default_rng(347);errors=[]
 for sample in range(4):
  c=s.coefficients;dm=np.array([c[:,:no]@c[:,:no].T]*2);q=rng.normal(size=dm.shape);q=.02*(q+q.transpose(0,2,1));dm+=q
  f,e=ctx.fock_and_energy(dm);fo,eo=original.fock_and_energy(dm);dp=rng.normal(size=(n,n));dp=.5*(dp+dp.T);dc=ctx.density_response_kernel(s)(dp);do=original.density_response_kernel(s)(dp);sp=np.array([dp,-dp]);sc=ctx.polarization.response_batch(sp[None])[0];so=original.polarization.response_batch(sp[None])[0]
  ctx.energy_cache.clear();energy_only_error=max(abs(ctx.energy(dm)-e),abs(base.energy(dm)-base.fock_and_energy(dm)[1]))
  errors.append(dict(energy=abs(e-eo),energy_only=energy_only_error,Fock=float(np.max(abs(f-fo))),charge_response=float(np.max(abs(dc-do))),spin_response=float(np.max(abs(sc-so)))))
 sites=list(range(no-2,no+2));ctx.register_density_span(np.array([np.outer(s.coefficients[:,j],s.coefficients[:,j]) for j in sites]));span_error=0.
 for hole in sites[:2]:
  occupations=np.zeros((2,n));occupations[:,:no]=1;occupations[0,hole]=0;occupations[0,no]=1;dm=np.array([(s.coefficients*o)@s.coefficients.T for o in occupations]);ctx.energy_cache.clear();span_error=max(span_error,abs(ctx.energy(dm)-ctx.fock_and_energy(dm)[1]))
 assert span_error<1e-10
 a=old.arrays;matrices=[]
 for context,response in [(original,r),(ctx,reference)]:
  engine=MatrixResponse(context,response,a['ground'],a['states'],a['multipliers'],a['auxiliary_frame'],old.labels,a['h'],a['eri'],derivatives=False,auxiliary_records=old.auxiliary_records);hh,ss,eg=engine.matrices(np.zeros(0));matrices.append((hh.v,ss.v))
 matrix_error=max(float(np.max(abs(a-b))) for a,b in zip(*matrices));ground=a['ground'];c=a['states'][0];transition=np.outer(ground[:,:no].T@s.overlap@c[:,0],ground[:,no:].T@s.overlap@c[:,no]);transition/=np.linalg.norm(transition);rotation=rng.normal(size=(n,n));rotation-=rotation.T;rotation*=.006/np.linalg.norm(rotation)
 fast=MatrixResponse(ctx,reference,a['ground'],a['states'],a['multipliers'],a['auxiliary_frame'],old.labels,a['h'],a['eri'],derivatives=False,auxiliary_records=old.auxiliary_records,fixed_bra=True);hh,ss,eg=fast.matrices(np.zeros(0));fixed_error=max(float(np.max(abs(hh.v-matrices[1][0]))),float(np.max(abs(ss.v-matrices[1][1]))));assert fixed_error<1e-9
 coefficient,multiplier,solver=solve_roks(ctx,c@expm(rotation),ground,transition,a['multipliers'][0],tolerance=1e-11,linearization_context=base,exact_refinement=True)
 try:ctx.analytic_gradient(None,None)
 except NotImplementedError:pass
 else:raise AssertionError('vertical quadrature must reject nuclear gradient requests')
 assert ctx.polarization.vertical_only
 report=dict(host=socket.getfqdn(),molecule=name,errors=errors,registered_span_energy_max_error=span_error,spin_pure_CI_matrix_max_error=matrix_error,fixed_bra_CI_matrix_max_error=fixed_error,RI_preconditioned_solver=solver,RI_diagnostic=certificate,RI_role='preconditioner only; full quadrature restores exact XC energy and residual',nuclear_derivatives_supported=False,nuclear_gradient_rejection_verified=True)
 write(root/'certificate.json',report);assert max(max(row.values()) for row in errors)<1e-8 and matrix_error<1e-9;print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
 from threadpoolctl import threadpool_limits
 with threadpool_limits(2):run()
