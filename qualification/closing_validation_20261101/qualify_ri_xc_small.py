"""Measure RI-XC as a numerical approximation to the existing OO-CID surface."""
from prepare_fresh import *
from pigment_ri_xc import prepare as auxiliary_response
from pigment_compact_reference import Compact
from types import SimpleNamespace
from dataclasses import replace
from scipy.linalg import solve_sylvester
from pyscf import df,lib
from matrix_gfn2.nosi_surface import PreparedNOSIModel,save_nosi_result,load_nosi_result
from matrix_gfn2.spin_reference import save_response_reference
import shutil

def run(cohort,name,hessian=False):
 root=HERE/'ri_xc_small'/name;root.mkdir(parents=True,exist_ok=True);case=prepare(cohort,name);s=case['s'];n=len(s.overlap);i,j=np.tril_indices(n);w=np.where(i==j,1.,2**.5);nf=len(i);E=np.zeros((nf,n,n));E[np.arange(nf),i,j]=1/w;E[np.arange(nf),j,i]=1/w
 # Same full geometry and reference density used by the established response.
 with np.load(CAM/cohort/name/'embedded_ground.npz') as z:p=z['density'].copy()
 mf=dft.RKS(case['mol']);mf.xc='CAM-B3LYP';mf.verbose=0;mf.grids.level=3;mf.small_rho_cutoff=0.;mf.grids.build(with_non0tab=True)
 full=df.DF(case['mol']);full.auxbasis='def2-universal-jkfit';full.build();factors=[]
 for block in full.loop(blksize=16):factors.append(ao2mo._ao2mo.nr_e2(block,np.asfortranarray(case['lift']),(0,n,0,n),aosym='s2',mosym='s1').reshape(len(block),n,n))
 factors=np.concatenate(factors);grid=root/'grid';grid.mkdir(exist_ok=True);paths=[];exact=[np.zeros((nf,nf)),np.zeros((nf,nf))]
 for block,(ao,mask,weight,coordinates) in enumerate(mf._numint.block_loop(case['mol'],mf.grids,case['mol'].nao_nr(),deriv=1,max_memory=128)):
  phi=ao@case['lift'];core=mf._numint.eval_rho(case['mol'],ao,case['core_density'],mask,'GGA',hermi=1);path=grid/f'chunk_{block:04d}.npz';np.savez(path,phi=phi,core=core,weight=weight,coordinates=coordinates);paths.append(path);rho=.5*(core+Compact.rho(phi,p));fxc=mf._numint.eval_xc_eff('CAM-B3LYP',np.array([rho,rho]),deriv=2,xctype='GGA')[2];fc=.25*(fxc[0,:,0]+fxc[0,:,1]+fxc[1,:,0]+fxc[1,:,1]);fs=.25*(fxc[0,:,0]-fxc[0,:,1]-fxc[1,:,0]+fxc[1,:,1])
  for lo in range(0,len(weight),512):
   hi=min(lo+512,len(weight));ph=phi[:,lo:hi];f=np.empty((4,nf,hi-lo));f[0]=w[:,None]*ph[0,:,i]*ph[0,:,j]
   for k in range(1,4):f[k]=w[:,None]*(ph[k,:,i]*ph[0,:,j]+ph[0,:,i]*ph[k,:,j])
   flat=f.transpose(1,0,2).reshape(nf,-1)
   for channel,fx in enumerate([fc[:,:,lo:hi],fs[:,:,lo:hi]]):exact[channel]+=flat@(np.einsum('ijg,jag->aig',fx,f,optimize=True)*weight[None,None,lo:hi]).reshape(nf,-1).T
 small=SimpleNamespace(name=name,n=n,mol=case['mol'],lift=case['lift'],mf=mf,a=factors,grid=paths,rho=Compact.rho)
 fitted,certificate=auxiliary_response(small,p,i,j,w,root)
 old=PreparedNOSIModel(HERE/'disentangled_bath_spectral_block_reference_eta_0.600000'/cohort/name/'model.json');operators=np.array([2*solve_sylvester(s.overlap,s.overlap,e) for e in E]);T=np.linalg.lstsq(old.response.operators.reshape(nf,-1).T,operators.reshape(nf,-1).T,rcond=None)[0].T
 response=replace(old.response,pair_charge_hartree=old.response.pair_charge_hartree+T.T@(fitted[0]-exact[0])@T,pair_spin_hartree=old.response.pair_spin_hartree+T.T@(fitted[1]-exact[1])@T,source_model=old.response.source_model+':RI_XC_control')
 target=root/'model';target.mkdir(exist_ok=True);save_response_reference(target/'response.npz',response)
 for filename in ['template.npz','arrays.npz']:shutil.copy2(old.root/filename,target/filename)
 manifest=dict(old.manifest);manifest['sha256']={f:sha(target/f) for f in ['response.npz','template.npz','arrays.npz']};write(target/'model.json',manifest);new=PreparedNOSIModel(target/'model.json');properties=('energy','gradient','hessian') if hessian else ('energy',);value=new.evaluate(case['calc'],case['xyz'],properties=properties);save_nosi_result(target/'checkpoint.npz',value);baseline=load_nosi_result(old.root/'checkpoint.npz',old);errors={a.state_id:(b.excitation_hartree-a.excitation_hartree)*EV for a,b in zip(baseline.states,value.states)}
 report=dict(molecule=name,parent_model=old.fingerprint,model=new.fingerprint,RI_XC=certificate,excitation_changes_ev=errors,maximum_change_ev=max(abs(x) for x in errors.values()),energy_labels_read=False)
 if hessian:
  previous=load_nosi_result(old.root/'analytic_audit/hessian.npz',old);report['gradient_max_change']=max(float(np.max(abs(a.gradient_hartree_per_bohr-b.gradient_hartree_per_bohr))) for a,b in zip(previous.states,value.states));report['Hessian_max_change']=max(float(np.max(abs(a.hessian_hartree_per_bohr2-b.hessian_hartree_per_bohr2))) for a,b in zip(previous.states,value.states))
 write(root/'certificate.json',report);assert report['maximum_change_ev']<.005;print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('cohort');parser.add_argument('name');parser.add_argument('--hessian',action='store_true');a=parser.parse_args()
 from threadpoolctl import threadpool_limits
 with threadpool_limits(2):run(a.cohort,a.name,a.hessian)
