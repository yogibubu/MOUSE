"""Same DFT spectral correction for TDA and ROKS; cached electronic references."""
from remote_bootstrap import HOST
from pathlib import Path
from types import SimpleNamespace
import argparse,json,hashlib,time,numpy as np
from scipy.optimize import linear_sum_assignment
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.hybrid_spectrum import match_orbital_spectrum,replace_tda_reference_gaps,correct_roks_excitation,with_occupied_extent_scaling
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_gfn2.stda import compact_xtb_tda,closed_shell_stda_matrix_derivative
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
parser=argparse.ArgumentParser()
parser.add_argument('--reference-energy-scale',type=float,required=True)
parser.add_argument('--extent-policy',choices=['none','basis_compactness'],default='none')
parser.add_argument('--extent-power',type=float,default=1.)
args=parser.parse_args()
if not np.isfinite(args.reference_energy_scale) or args.reference_energy_scale<=0:
 parser.error('reference energy scale must be finite and positive')
if not np.isfinite(args.extent_power) or args.extent_power<=0:
 parser.error('extent power must be finite and positive')
if args.extent_policy=='none' and args.extent_power!=1.:
 parser.error('extent power requires an explicit extent policy')
RUN=OUT/(f'scale_{args.reference_energy_scale:.12g}'+('_occupied_extent'+(f'_p{args.extent_power:.12g}' if args.extent_power!=1. else '') if args.extent_policy!='none' else ''));RUN.mkdir(exist_ok=True)
PREVIOUS=ROOT.parent/'uracil_uridine_20261029';EV=27.211386245988
DFTROOT=Path('/home/enzo/MATRIX-runs/tda-uracil-application-20261022-v1/docs/reports/tda_uracil_application_20261022/results')
def write(path,data):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else str(a))+'\n');t.replace(path)
meta=json.loads((PREVIOUS/'inputs/quest_s0.json').read_text());xyz=np.array(meta['coordinates_angstrom']);gs=load_derivative_point(PREVIOUS/'results/quest_s0/ground.npz',request_hash='quest_s0',coordinates_angstrom=xyz);s=gs.reference_state
calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[z] for z in s.atomic_numbers],temperature_kelvin=0.)
occ=np.flatnonzero(s.alpha_occupations>.5);virt=np.flatnonzero(s.alpha_occupations<.5);ii=np.repeat(occ,len(virt));aa=np.tile(virt,len(occ));ci=s.coefficients[:,ii];ca=s.coefficients[:,aa]
cache=OUT/'source_matrices.npz'
if cache.exists():
 with np.load(cache) as f:parent=f['parent'];empirical=f['empirical']
else:
 kernel=calc.density_response_kernel(s);cols=[]
 for i,a in zip(ii,aa):
  tr=np.outer(s.coefficients[:,i],s.coefficients[:,a]);f=kernel(.5*(tr+tr.T));cols.append(np.einsum('mi,mn,ni->i',ci,f,ca))
 parent=np.diag(s.orbital_energies_hartree[aa]-s.orbital_energies_hartree[ii])+2*np.array(cols).T
 selected=compact_xtb_tda(s,roots=10,virtual_count=len(virt),shift_model='constant')
 # Configuration order is an explicit rectangle, checked against the exported selection.
 assert np.array_equal(ii,selected.occupied_orbitals) and np.array_equal(aa,selected.virtual_orbitals)
 shape=xyz.shape;matrix=closed_shell_stda_matrix_derivative(s,selected,np.zeros((*shape,s.nao,s.nao)),np.zeros((*shape,s.nao,s.nmo)),np.zeros((*shape,s.nmo)),spectral_range_ev=10.)
 empirical=matrix.matrix_hartree
 np.savez_compressed(cache,parent=parent,empirical=empirical,occupied=ii,virtual=aa)
for a in (parent,empirical):assert np.max(abs(a-a.T))<1e-10
pvals,pvectors=np.linalg.eigh(parent);evals,evectors=np.linalg.eigh(empirical)
# Seed n/pi identity from the individually inspected canonical promotions.
targets=[('n_pi',20,21,1),('pi_pi',19,21,2)]
anchors={}
for kernel,matrix,vals,vecs in [('parent',parent,pvals,pvectors),('empirical',empirical,evals,evectors)]:
 for target,hole,particle,_ in targets:
  pair=np.flatnonzero((ii==hole)&(aa==particle))[0];root=int(np.argmax(abs(vecs[pair])**2));anchors[kernel,target]=root
ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=SimpleNamespace(converged=True,orbital_state=s))
roks={};decomp={}
for target,hole,particle,index in targets:
 folder=PREVIOUS/f'results/quest_s0/gmf_{target}_{index}_ground';d=json.loads((folder/'summary.json').read_text());r=load_derivative_point(folder/'converged.npz',request_hash=d['request_hash'],coordinates_angstrom=xyz);roks[target]=r
 pm=np.array([(s.coefficients*o)@s.coefficients.T for o in r.occupations]);pt=np.array([(s.coefficients*o)@s.coefficients.T for o in r.triplet_occupations]);_,em=ctx.fock_and_energy(pm);_,et=ctx.fock_and_energy(pt)
 frozen=2*em-et-r.ground_energy_hartree;gap=s.orbital_energies_hartree[particle]-s.orbital_energies_hartree[hole]
 decomp[target]=dict(se_gap_ev=gap*EV,se_frozen_excitation_ev=frozen*EV,se_frozen_beyond_gap_ev=(frozen-gap)*EV,se_relaxation_ev=(r.excitation_energy_hartree-frozen)*EV,se_roks_ev=r.excitation_energy_hartree*EV)
records=[]
for name in ['b3lyp_avdz','pbe0_avdz','pbe0_avtz']:
 refpath=DFTROOT/name/'reference.npz'
 with np.load(refpath) as d:
  assert np.array_equal(d['atomic_numbers'],s.atomic_numbers)
  assert np.max(abs(d['coordinates_bohr']-s.coordinates_bohr))<1e-8
  ref=dict(c=d['coefficients'],s=d['overlap'],o=d['occupations'],e=d['orbital_energies_hartree'])
 with np.load(OUT/f'{name}_mapping_probe.npz') as d:cross=d['cross_overlap'].T
 mapping=match_orbital_spectrum(s.coefficients,s.overlap,s.alpha_occupations+s.beta_occupations,s.orbital_energies_hartree,ref['c'],ref['s'],ref['o'],ref['e'],cross,reference_model=name,reference_energy_scale=args.reference_energy_scale)
 if args.extent_policy=='basis_compactness':
  with np.load(OUT/'extent/gfn2_moments.npz') as f:source_radii=f['radii']
  with np.load(OUT/f'extent/{name}_moments.npz') as f:
   assert np.array_equal(f['reference_indices'],mapping.reference_indices)
   reference_radii=f['mapped_radii']
  mapping=with_occupied_extent_scaling(mapping,source_radii,reference_radii,power=args.extent_power)
 good=mapping.qualified[ii]&mapping.qualified[aa];active=np.flatnonzero(good)
 np.savez_compressed(RUN/f'{name}_mapped_spectrum.npz',source_energies=mapping.source_energies_hartree,reference_energies=mapping.reference_energies_hartree,effective_reference_energies=mapping.effective_reference_energies_hartree,reference_energy_scale=mapping.reference_energy_scale,extent_power=args.extent_power,extent_policy=args.extent_policy,reference_indices=mapping.reference_indices,overlaps=mapping.overlaps,margins=mapping.margins,qualified=mapping.qualified)
 record=dict(reference=name,reference_energy_scale=mapping.reference_energy_scale,kernel_scaling='unchanged from cached semiempirical TDA matrix',reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),source_fingerprint=mapping.source_fingerprint,occupied_count=len(occ),virtual_count=len(virt),full_configurations=len(ii),trusted_configurations=len(active),minimum_overlap=.7,minimum_margin=.05,targets=[],tda=[])
 record.update(extent_policy=args.extent_policy,extent_power=args.extent_power,effective_reference_energies_hartree=mapping.effective_reference_energies_hartree.tolist(),extent_weights=(mapping.extent_scaling.weights.tolist() if mapping.extent_scaling is not None else None))
 for target,hole,particle,index in targets:
  corrected=correct_roks_excitation(roks[target],mapping,hole,particle)
  record['targets'].append(dict(state=target,hole=hole,particle=particle,dft_hole=int(mapping.reference_indices[hole]),dft_particle=int(mapping.reference_indices[particle]),hole_overlap=mapping.overlaps[hole],particle_overlap=mapping.overlaps[particle],hole_margin=mapping.margins[hole],particle_margin=mapping.margins[particle],dft_gap_ev=(mapping.reference_energies_hartree[particle]-mapping.reference_energies_hartree[hole])*EV,correction_ev=corrected.gap_correction_hartree*EV,hybrid_roks_ev=corrected.excitation_energy_hartree*EV,**decomp[target]))
 for kernel,a,basevals,basevectors in [('parent',parent,pvals,pvectors),('empirical',empirical,evals,evectors)]:
  for space,ids,strict in [('trusted',active,True),('full_diagnostic',np.arange(len(ii)),False)]:
   orig=a[np.ix_(ids,ids)];corrected=replace_tda_reference_gaps(orig,ii[ids],aa[ids],mapping,require_qualified=strict);values,vectors=np.linalg.eigh(corrected);origvalues,origvectors=np.linalg.eigh(orig)
   fullvectors=np.zeros((len(ii),len(values)));fullvectors[ids]=vectors
   scores=np.array([abs(basevectors[:,anchors[kernel,t]]@fullvectors) for t,_,_,_ in targets]);rows,columns=linear_sum_assignment(-scores**2)
   for row,col in zip(rows,columns):
    target=targets[row][0];confidence=float(scores[row,col]);margin=confidence-float(np.max(np.delete(scores[row],col),initial=0.));trustedweight=float(np.sum(fullvectors[good,col]**2))
    source_root=anchors[kernel,target]
    base_scores=abs(basevectors[:,source_root]@np.eye(len(ii))[:,ids]@origvectors);base_root=int(np.argmax(base_scores))
    record['tda'].append(dict(kernel=kernel,space=space,state=target,root=int(col+1),hybrid_tda_ev=float(values[col]*EV),source_full_tda_ev=float(basevals[source_root]*EV),source_same_space_tda_ev=float(origvalues[base_root]*EV),truncation_error_ev=float((origvalues[base_root]-basevals[source_root])*EV),state_overlap=confidence,margin=margin,trusted_configuration_weight=trustedweight,state_mapping_qualified=bool(confidence>=.7 and margin>=.05 and trustedweight>=.95)))
   np.savez_compressed(RUN/f'{name}_{kernel}_{space}_tda.npz',energies=values,amplitudes=fullvectors.T,occupied=ii,virtual=aa,configuration_indices=ids)
 records.append(record);write(RUN/f'{name}_comparison.json',record)
write(RUN/'comparison.json',dict(host=HOST,reference_energy_scale=args.reference_energy_scale,geometry='exact QUEST uracil',ground_reference='same cached GFN2 canonical reference for all methods',new_scf_jobs=0,references=records))
for record in records:
 print(record['reference'],'trusted configs',record['trusted_configurations'],flush=True)
 for row in record['targets']:print('ROKS',row['state'],row['hybrid_roks_ev'],'correction',row['correction_ev'],flush=True)
 for row in record['tda']:
  if row['space']=='trusted':print('TDA',row,flush=True)
