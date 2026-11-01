"""Leading retained-pole normalization of the frozen correlation operator.

The same 50/50 F/Q self-energy has derivative Sigma'_common=.5 Sigma'_F.
Its diagonal quasiparticle residues are Z_i=[1+.5||tail_i||^2]^-1.
Apply sqrt(Z_i) to each external leg of the screened interaction and normal
order against the corrected native ground Fock. This is a diagonal coherent-
pole approximation, not the exact two-electron downfolding with bath vertices.
Unqualified channels keep Z=1, just as they keep their uncorrected one-body
reference. No excitation label or tunable coefficient is an input.
"""
from prepare_fresh import *
from matrix_gfn2.nosi_surface import PreparedNOSIModel,load_nosi_result,save_nosi_result
import shutil

def run(cohort,name):
 parent=HERE/'downfold_reference'/cohort/name;out=HERE/'pole_residue_screening'/cohort/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 model=PreparedNOSIModel(parent/'model.json');r=load_nosi_result(parent/'checkpoint.npz',model);case=prepare(cohort,name);s=case['s'];no=case['no'];sr=model._anchor_sqrt;si=np.linalg.inv(sr);report=json.loads((parent/'result.json').read_text());good=np.array(report['qualified'],bool);z=np.ones(len(good));zf=np.array(report['retained_scalar_residue']);z[good]=1/(1+.5*(1/zf[good]-1));c=sr@s.coefficients;leg=(c*np.sqrt(z))@c.T
 w=np.einsum('pqrs,pi,qj,rk,sl->ijkl',model.arrays['eri'],leg,leg,leg,leg,optimize=True)
 ctx=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=model.anchor,spin_reference=model.response);p=r.ground_coefficients[:,:no]@r.ground_coefficients[:,:no].T;fg,_=ctx.fock_and_energy(np.array([p,p]));pg=2*sr@p@sr;h=si@fg[0]@si-np.einsum('pqrs,rs->pq',w,pg)+.5*np.einsum('prqs,rs->pq',w,pg)
 for filename in ['template.npz','response.npz']:shutil.copy2(parent/filename,out/filename)
 arrays=dict(model.arrays);arrays.update(h=h,eri=w,ground=r.ground_coefficients,states=r.state_coefficients,multipliers=r.multipliers,reference_ci=r.ci_coefficients);np.savez_compressed(out/'arrays.npz',**arrays);manifest=dict(model.manifest);manifest['sha256']={f:sha(out/f) for f in ['template.npz','response.npz','arrays.npz']};manifest['provenance']=dict(manifest['provenance'],correlation_operator='diagonal coherent-pole external-leg normalization',residues_common=z,parent_model=model.fingerprint,approximation='diagonal residues only; bath two-electron vertices neglected');write(out/'model.json',manifest)
 final=PreparedNOSIModel(out/'model.json');result=final.evaluate(case['calc'],case['xyz'],properties=('energy',));save_nosi_result(out/'checkpoint.npz',result)
 write(out/'result.json',dict(cohort=cohort,molecule=name,states=[dict(id=v.state_id,gap_ev=v.excitation_hartree*EV,overlap=v.overlap,margin=v.margin) for v in result.states],model=final.fingerprint,weights=[.5,.5],lambda_exchange=.9,offset_ev=0,reference_excitation_energies_read=False,residues_common=z,approximation='diagonal coherent-pole external-leg normalization of frozen screened operator'))
 print(name,[(v.state_id,v.excitation_hartree*EV) for v in result.states],flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('cohort');parser.add_argument('name');a=parser.parse_args()
 with threadpool_limits(2):run(a.cohort,a.name)
