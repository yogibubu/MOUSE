"""Frozen parameter-free law tested on thymine, with persistent checkpoints."""
from remote_bootstrap import HOST
from extent_support import source_molecule,radii
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import argparse,json,sys,time,hashlib,traceback,numpy as np
from pyscf import gto,scf,lib
from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_deltascf,evaluate_gfn2_roks_singlet
from matrix_gfn2.response_reference import stationary_response_reference
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.roks import ROKSSingletResult
from matrix_gfn2.hybrid_spectrum import match_orbital_spectrum,with_occupied_extent_scaling,correct_roks_excitation,replace_tda_reference_gaps
from matrix_apoc.deltascf import singlet_csf_overlap
lib.num_threads(2)
parser=argparse.ArgumentParser()
parser.add_argument('--reference-energy-scale',type=float,default=.9)
parser.add_argument('--extent-power',type=float,default=1.)
args=parser.parse_args()
if not all(np.isfinite(v) and v>0 for v in [args.reference_energy_scale,args.extent_power]):parser.error('scale and power must be finite and positive')
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/extent/thymine';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT.parent/'uracil_uridine_20261029'))
from benchmark import characterize
EV=27.211386245988
REFROOT=Path('/home/enzo/MATRIX-runs/tda-frozen-validation-20261022-v1/docs/reports/tda_frozen_validation_20261022/thymine')
def write(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n');tmp.replace(path)
with np.load(REFROOT/'results/pbe0_avdz/reference.npz') as f:
 xyz=f['coordinates_bohr']*.529177210903;zs=f['atomic_numbers']
calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in zs],temperature_kelvin=0.)
checkpoint=OUT/'ground.npz';key='extent_thymine_ground'
if checkpoint.exists():
 gs=load_derivative_point(checkpoint,request_hash=key,coordinates_angstrom=xyz);s=gs.reference_state
else:
 ground,cert=stationary_response_reference(calc,xyz);s=ground.orbital_state
 gs=evaluate_gfn2_deltascf(calc,xyz,s.alpha_occupations,s.beta_occupations,ground_state=ground,selection='aufbau',gradient=True,density_tolerance=1e-10,orbital_tolerance=1e-10,energy_tolerance=1e-12)
 save_derivative_point(checkpoint,gs,request_hash=key,coordinates_angstrom=xyz);s=gs.reference_state;write(OUT/'ground_certificate.json',cert)
ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=gs.energy_hartree)
ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground)
base=list(range(len(zs)));p,n,_,_=characterize(s,base)
occupied=np.flatnonzero(s.alpha_occupations>.5);virtual=np.flatnonzero(s.alpha_occupations<.5)
nh=max(i for i in occupied if n[i]>.3 and p[i]<.3);ph=max(i for i in occupied if p[i]>.4);particle=int(virtual[0]);assert p[particle]>.35
write(OUT/'seeds.json',dict(host=HOST,hole_n=int(nh),hole_pi=int(ph),particle=particle,pi_character=p,oxygen_character=n))
solutions={};status=[]
for name,hole,index in [('n_pi',nh,1),('pi_pi',ph,2)]:
 file=OUT/f'{name}.npz';key=f'extent_thymine_{name}';occ=np.array([s.alpha_occupations,s.beta_occupations]);occ[0,hole]=0;occ[0,particle]=1;t0=time.perf_counter()
 if file.exists():r=load_derivative_point(file,request_hash=key,coordinates_angstrom=xyz)
 else:
  c=s.coefficients.copy();latest=OUT/f'{name}_latest.npz'
  if latest.exists():
   with np.load(latest) as f:c=f['coefficients'];assert np.array_equal(f['occupations'],occ)
  triplet=np.array([np.maximum(*occ),np.minimum(*occ)]);pm=np.array([(c*o)@c.T for o in occ]);pt=np.array([(c*o)@c.T for o in triplet]);fm,em=ctx.fock_and_energy(pm);ft,et=ctx.fock_and_energy(pt)
  seed=ROKSSingletResult(c,occ,triplet,s.overlap,fm,ft,2*em-et,1.,0,False,reference_state=s,ground_energy_hartree=gs.energy_hartree,core_electrons=np.asarray(s.atomic_numbers)-np.asarray(calc._valence,dtype=int))
  def checkpoint_step(point):
   np.savez_compressed(latest,coefficients=point.point,occupations=occ,coordinates_angstrom=xyz);write(OUT/f'{name}_history.json',list(point.history))
  try:
   r=evaluate_gfn2_roks_singlet(calc,xyz,*occ,ground_state=ground,reference_result=seed,minimum_state_overlap=0.,gradient=True,orbital_solver='gmf',saddle_index=index,maximum_iterations=80,maximum_step=.2,orbital_tolerance=1e-9,freeze_and_release=True,freeze_iterations=8,checkpoint=checkpoint_step)
   save_derivative_point(file,r,request_hash=key,coordinates_angstrom=xyz)
  except Exception as exc:
   status.append(dict(state=name,error=str(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-t0));write(OUT/'status.json',dict(host=HOST,states=status));print(status[-1],flush=True);continue
 pp,nn,_,_=characterize(s,base,r.common_coefficients)
 csf_overlap,_=singlet_csf_overlap(replace(r,common_coefficients=s.coefficients).to_apoc(),r.to_apoc(),s.overlap)
 character_ok=bool(csf_overlap>=.7 and pp[particle]>.35 and (nn[hole]>.3 and pp[hole]<.3 if name=='n_pi' else pp[hole]>.4))
 orbital_overlap=float(abs(s.coefficients[:,hole]@s.overlap@r.common_coefficients[:,hole]))
 status.append(dict(state=name,hole=int(hole),particle=particle,se_gap_ev=r.excitation_energy_hartree*EV,index=r.orbital_saddle_index,residual=r.orbital_residual,character_qualified=character_ok,canonical_csf_overlap=csf_overlap,hole_pi=float(pp[hole]),particle_pi=float(pp[particle]),hole_n=float(nn[hole]),hole_overlap=orbital_overlap,seconds=time.perf_counter()-t0))
 if character_ok:solutions[name]=(hole,r)
 write(OUT/'status.json',dict(host=HOST,states=status));print(status[-1],flush=True)
small,t=source_molecule(s);rs=radii(small,t@s.coefficients)
ii=np.repeat(occupied,len(virtual));aa=np.tile(virtual,len(occupied));ci=s.coefficients[:,ii];ca=s.coefficients[:,aa]
cache=OUT/'parent_tda.npz'
if cache.exists():
 with np.load(cache) as f:A=f['matrix']
else:
 kernel=calc.density_response_kernel(s);columns=[]
 for i,a in zip(ii,aa):
  tr=np.outer(s.coefficients[:,i],s.coefficients[:,a]);fock=kernel(.5*(tr+tr.T));columns.append(np.einsum('mi,mn,ni->i',ci,fock,ca))
 A=np.diag(s.orbital_energies_hartree[aa]-s.orbital_energies_hartree[ii])+2*np.array(columns).T;np.savez_compressed(cache,matrix=A,occupied=ii,virtual=aa)
basev,basex=np.linalg.eigh(A)
records=[]
for name in ['pbe0_avdz','pbe0_avtz']:
 folder=REFROOT/'results'/name;mol=scf.chkfile.load_mol(str(folder/'backend.chk'))
 with np.load(folder/'reference.npz') as f:
  assert np.array_equal(f['atomic_numbers'],s.atomic_numbers);assert np.max(abs(f['coordinates_bohr']-s.coordinates_bohr))<1e-8
  cr=f['coefficients'];er=f['orbital_energies_hartree'];orr=f['occupations'];sr=f['overlap']
 cross=(gto.intor_cross('int1e_ovlp',mol,small)@t).T
 mapping=match_orbital_spectrum(s.coefficients,s.overlap,s.alpha_occupations+s.beta_occupations,s.orbital_energies_hartree,cr,sr,orr,er,cross,reference_model=f'thymine_{name}',reference_energy_scale=args.reference_energy_scale)
 rd=radii(mol,cr)[mapping.reference_indices];extent=with_occupied_extent_scaling(mapping,rs,rd,power=args.extent_power)
 allowed=mapping.qualified[ii]&mapping.qualified[aa];ids=np.flatnonzero(allowed)
 for policy,m in [(f'constant_{args.reference_energy_scale:.12g}',mapping),('basis_compactness',extent)]:
  values,vectors=np.linalg.eigh(replace_tda_reference_gaps(A[np.ix_(ids,ids)],ii[ids],aa[ids],m));embed=np.zeros((len(ii),len(values)));embed[ids]=vectors
  for state,(hole,r) in solutions.items():
   try:corrected=correct_roks_excitation(r,m,hole,particle)
   except ValueError as exc:records.append(dict(reference=name,policy=policy,state=state,error=str(exc)));continue
   pair=np.flatnonzero((ii==hole)&(aa==particle))[0];anchor=int(np.argmax(basex[pair]**2));score=abs(basex[:,anchor]@embed);root=int(np.argmax(score));tbe=4.98 if state=='n_pi' else 5.279
   records.append(dict(reference=name,policy=policy,state=state,se_roks_ev=r.excitation_energy_hartree*EV,se_orbital_gap_ev=float((s.orbital_energies_hartree[particle]-s.orbital_energies_hartree[hole])*EV),dft_orbital_gap_ev=float((m.reference_energies_hartree[particle]-m.reference_energies_hartree[hole])*EV),roks_ev=corrected.excitation_energy_hartree*EV,roks_error_ev=corrected.excitation_energy_hartree*EV-tbe,tbe_ev=tbe,tda_ev=float(values[root]*EV),tda_state_overlap=float(score[root]),trusted_configurations=len(ids),hole_overlap=float(m.overlaps[hole]),particle_overlap=float(m.overlaps[particle]),source_hole_radius_bohr=float(rs[hole]),reference_hole_radius_bohr=float(rd[hole]),lambda_hole=float(args.reference_energy_scale if m.extent_scaling is None else args.reference_energy_scale*(rs[hole]/rd[hole])**args.extent_power)))
output='comparison.json' if args.reference_energy_scale==.9 and args.extent_power==1. else f'comparison_scale_{args.reference_energy_scale:.12g}_p{args.extent_power:.12g}.json'
write(OUT/output,dict(host=HOST,model_fixed_before_holdout=True,reference_energy_scale=args.reference_energy_scale,extent_power=args.extent_power,states=status,records=records,reference_protocol_sha256=hashlib.sha256((REFROOT/'protocol.json').read_bytes()).hexdigest()))
for row in records:print(row,flush=True)
