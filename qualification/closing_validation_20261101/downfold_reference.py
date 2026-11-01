"""Diagnose valence-basis confinement by parameter-free Feshbach self-energy.

At each already qualified DFT orbital energy, eliminate the orthogonal
complement of the full native valence embedding. Add its diagonal self-energy
to the F reference, retain the existing Q reference and their 50/50 average.
No excitation energy, lambda change, fitted coefficient or offset is read.
The first diagnostic holds the screened two-body operator fixed to isolate
the one-body reference error; it is not silently substituted for the frozen
prospective test.
"""
from prepare_fresh import *
from scipy.linalg import null_space
from dataclasses import replace
from matrix_gfn2.nosi_surface import PreparedNOSIModel,save_nosi_result
from matrix_gfn2.spin_reference import save_response_reference
import shutil

def corrected(cohort,name):
 out=HERE/'downfold_reference'/cohort/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 original=PreparedNOSIModel(REPAIR/'prepared_models'/cohort/name/'model.json');case=prepare(cohort,name);s=case['s'];c=s.coefficients;no=case['no'];folder=CAM/cohort/name
 certificate=json.loads((NATIVE/cohort/name/'coherent_common/reference_certificate.json').read_text());good=np.array(certificate['qualified_mapping'],bool);mapping=np.array(certificate['mapping'],int)
 with np.load(folder/'CAM_full_reference/reference.npz') as z:cd=z['coefficients'].copy();od=z['occupations'].copy();ed=z['orbital_energies'].copy()
 nc=np.count_nonzero(od==2)-no;selected=mapping
 # The embedding is orthogonal to the archived frozen PBE0 core, not
 # exactly to the separately optimized CAM core. Retain the complete
 # CAM eigenbasis when forming its orthogonal complement.
 # Native embedding columns are orthonormal in the DFT MO metric.
 projection=cd.T@case['sd']@case['lift']@c;assert np.max(abs(projection.T@projection-np.eye(len(c))))<1e-8
 complement=null_space(projection.T);hpp=projection.T@(ed[:,None]*projection);hpq=projection.T@(ed[:,None]*complement);hqq=complement.T@(ed[:,None]*complement);pole,uq=eigh(hqq);coupling=hpq@uq;self_energy=np.zeros(len(c));pole_distance=np.zeros(len(c));spectral_norm=np.zeros(len(c))
 for i in range(len(c)):
  if not good[i]:continue
  delta=ed[selected[i]]-pole;pole_distance[i]=np.min(abs(delta),initial=np.inf)
  # The coupling to an exactly decoupled complement pole contributes zero.
  use=abs(delta)>1e-10
  assert np.max(abs(coupling[i,~use]),initial=0)<1e-8,(name,i,'unresolved coupled complement pole')
  self_energy[i]=np.sum(coupling[i,use]**2/delta[use]);spectral_norm[i]=1/(1+np.sum(coupling[i,use]**2/delta[use]**2))
 delta_levels=.5*self_energy;delta_f=(s.overlap@c*delta_levels)@c.T@s.overlap;dual=np.array([2*solve_sylvester(s.overlap,s.overlap,a) for a in original.response.operators]);delta_linear=np.einsum('aij,ij->a',dual,delta_f)
 response=replace(original.response,pair_linear_hartree=original.response.pair_linear_hartree+delta_linear,source_model=original.response.source_model+':feshbach_complement_self_energy')
 save_response_reference(out/'response.npz',response);shutil.copy2(original.root/'template.npz',out/'template.npz');arrays=dict(original.arrays);delta_h=original._anchor_sqrt@c@np.diag(delta_levels)@c.T@original._anchor_sqrt;arrays['h']=arrays['h']+delta_h;np.savez_compressed(out/'arrays.npz',**arrays)
 manifest=dict(original.manifest);manifest['sha256']={filename:sha(out/filename) for filename in ['template.npz','response.npz','arrays.npz']};manifest['provenance']=dict(manifest['provenance'],parent_model=original.fingerprint,reference_correction='diagonal Feshbach self-energy of excluded full-DFT complement, evaluated at qualified matched orbital levels',fitted_parameters=0,screened_operator='held fixed as diagnostic',self_energy_hartree=self_energy,delta_common_levels_hartree=delta_levels);write(out/'model.json',manifest)
 model=PreparedNOSIModel(out/'model.json');calc=case['calc'];calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Unexpected SCC in downfolded prepared evaluation'))
 value=model.evaluate(calc,case['xyz'],properties=('energy',));save_nosi_result(out/'checkpoint.npz',value)
 write(out/'result.json',dict(cohort=cohort,molecule=name,model=model.fingerprint,parent_model=original.fingerprint,states=[dict(id=v.state_id,gap_ev=v.excitation_hartree*EV,overlap=v.overlap,margin=v.margin) for v in value.states],self_energy_hartree=self_energy,delta_common_levels_hartree=delta_levels,F_levels=certificate['F_levels'],Q_levels=certificate['Q_levels'],qualified=good,complement_pole_distance_hartree=pole_distance,retained_scalar_residue=spectral_norm,rayleigh_full_DFT=np.diag(hpp),native_valence_projection=sha(case['src']/'full_response/anchor/embedding.npz'),no_reference_excitation_energy_read=True,weights=[.5,.5],lambda_exchange=.9,offset_ev=0,DFT_backend_only_preparation=True))
 print(name,'self energy',self_energy,'gaps',[(v.state_id,v.excitation_hartree*EV) for v in value.states],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');a=p.parse_args()
 with threadpool_limits(2):corrected(a.cohort,a.name)
