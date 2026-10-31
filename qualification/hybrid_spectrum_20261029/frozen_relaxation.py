"""No-fit frozen DFT excitation plus retained semiempirical orbital relaxation."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,time,traceback,numpy as np
from pyscf import dft,scf,lib
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_gfn2.hybrid_spectrum import match_orbital_spectrum,spectrum_fingerprint
from extent_support import source_molecule
from generalization import inputs as generalization_inputs
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'frozen_relaxation';EV=27.211386245988
CONFIG=json.loads((OUT/'protocol.json').read_text());lib.num_threads(CONFIG['threads'])

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 tmp.write_text(json.dumps(data,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n');tmp.replace(path)
def key(data):return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
def cached(path,request):
 if not path.exists():return None
 data=json.loads(path.read_text())
 if data.get('request_hash')!=request:raise ValueError(f'cached frozen calculation differs from request: {path}')
 return data

def plan():
 cases={};previous=ROOT.parent/'uracil_uridine_20261029'
 dftroot=Path('/home/enzo/MATRIX-runs/tda-uracil-application-20261022-v1/docs/reports/tda_uracil_application_20261022/results/pbe0_avdz')
 uro=[]
 for state,h,p,i,tbe in [('n_pi',20,21,1,4.935),('pi_pi',19,21,2,5.381)]:
  folder=previous/f'results/quest_s0/gmf_{state}_{i}_ground';meta=json.loads((folder/'summary.json').read_text())
  uro.append(dict(id=state,hole=h,particle=p,tbe_ev=tbe,checkpoint=str(folder/'converged.npz'),request_hash=meta['request_hash']))
 cases['uracil']=dict(name='uracil',cohort='development',reference=str(dftroot),targets=uro)
 thy=ROOT/'results/extent/thymine';meta=json.loads((thy/'comparison.json').read_text())
 targets=[]
 for state,tbe in [('n_pi',4.98),('pi_pi',5.279)]:
  m=next(x for x in meta['states'] if x['state']==state);assert m['character_qualified'] and m['canonical_csf_overlap']>=.7
  targets.append(dict(id=state,hole=m['hole'],particle=m['particle'],tbe_ev=tbe,checkpoint=str(thy/f'{state}.npz'),request_hash=f'extent_thymine_{state}'))
 cases['thymine']=dict(name='thymine',cohort='development',reference='/home/enzo/MATRIX-runs/tda-frozen-validation-20261022-v1/docs/reports/tda_frozen_validation_20261022/thymine/results/pbe0_avdz',targets=targets)
 cohort=json.loads((ROOT/'generalization/summary.json').read_text());protocol=json.loads((ROOT/'generalization/protocol.json').read_text())
 for case in protocol['cases']:
  folder,_,_,_=generalization_inputs(case);targets=[]
  for row in [x for x in cohort['rows'] if x['molecule']==case['name']]:
   accepted=next((a for a in row['solver_attempts'] if a.get('state_qualified')),None)
   if accepted is None:
    targets.append(dict(id=row['id'],tbe_ev=row['tbe_ev'],prior_exclusion=row['exclusion_reason']));continue
   i=accepted['observed_index'];h=row['hole'];p=row['particle']
   targets.append(dict(id=row['id'],hole=h,particle=p,tbe_ev=row['tbe_ev'],checkpoint=str(ROOT/f'generalization/{case["name"]}/{row["id"]}_index_{i}/converged.npz'),request_hash=f'generalization_{case["name"]}_{row["id"]}_{h}_{p}_{i}'))
  cases[case['name']]=dict(name=case['name'],cohort='previous_generalization',reference=str(folder),targets=targets)
 return [cases[name] for name in CONFIG['planned_molecules']]

def state_from(target):
 path=Path(target['checkpoint'])
 with np.load(path) as f:xyz=f['coordinates_angstrom']
 r=load_derivative_point(path,request_hash=target['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
 if not r.converged or r.orbital_residual>1e-9 or r.spin_squared!=0:raise ValueError('unqualified retained ROKS state')
 return r,xyz

def run(case):
 name=case['name'];out=OUT/name;out.mkdir(exist_ok=True);folder=Path(case['reference'])
 reference_path=folder/'reference.npz';chk_path=folder/'backend.chk'
 refhash=digest(reference_path)
 with np.load(reference_path) as f:ref={k:f[k] for k in f.files}
 valid=[t for t in case['targets'] if 'checkpoint' in t]
 result0,xyz=state_from(valid[0]);s=result0.reference_state
 assert np.array_equal(ref['atomic_numbers'],s.atomic_numbers)
 assert np.max(abs(ref['coordinates_bohr']-s.coordinates_bohr))<1e-8
 refC=ref['coefficients'];S=ref['overlap'];occ=ref['occupations'];n=len(occ)
 assert np.allclose(refC.T@S@refC,np.eye(n),atol=2e-7,rtol=0)
 mol=scf.chkfile.load_mol(str(chk_path));mol.max_memory=CONFIG['max_memory_mb']
 assert np.max(abs(mol.intor('int1e_ovlp')-S))<1e-9
 small,transform=source_molecule(s)
 from pyscf import gto
 cross=(gto.intor_cross('int1e_ovlp',mol,small)@transform).T
 mapping=match_orbital_spectrum(s.coefficients,s.overlap,s.alpha_occupations+s.beta_occupations,s.orbital_energies_hartree,refC,S,occ,ref['orbital_energies_hartree'],cross,reference_model='PBE0/aug-cc-pVDZ/'+name)
 ref_fingerprint=mapping.source_fingerprint
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
 ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=result0.ground_energy_hartree)
 ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground)
 _,ground_se=ctx.fock_and_energy(np.array([(s.coefficients*o)@s.coefficients.T for o in [s.alpha_occupations,s.beta_occupations]]))
 assert abs(ground_se-result0.ground_energy_hartree)<1e-8
 source_hashes={name:digest(BASE/'source/packages/matrix-gfn2/src/matrix_gfn2'/name) for name in ['deltascf.py','roks.py','spin_polarization.py']}
 mf=None
 def setup():
  nonlocal mf
  if mf is None:
   mf=dft.UKS(mol);mf.xc='PBE0';mf.grids.level=CONFIG['grid_level'];mf.small_rho_cutoff=CONFIG['small_rho_cutoff'];mf.max_memory=CONFIG['max_memory_mb'];mf.verbose=0
   mf.grids.build(with_non0tab=True)
  return mf
 def energy(dm):
  mf=setup();electrons=np.einsum('smn,nm->s',dm,S)
  assert abs(electrons.sum()-np.sum(occ))<1e-7
  t0=time.perf_counter();v=mf.get_veff(mol,dm);e=float(mf.energy_tot(dm=dm,h1e=mf.get_hcore(),vhf=v))
  if not np.isfinite(e):raise ValueError('nonfinite frozen DFT energy')
  return dict(energy_hartree=e,alpha_electrons=float(electrons[0]),beta_electrons=float(electrons[1]),coulomb_hartree=float(v.ecoul),xc_including_hf_hartree=float(v.exc),seconds=time.perf_counter()-t0,grid_points=len(mf.grids.coords))
 def density(occupation):return np.array([(refC*o)@refC.T for o in occupation])
 parameters=dict(schema='frozen-relaxation-v1',reference_sha256=refhash,xc='PBE0',grid_level=CONFIG['grid_level'],small_rho_cutoff=CONFIG['small_rho_cutoff'])
 ground_key=key(parameters);gs_path=out/'dft_ground.json';gs=cached(gs_path,ground_key)
 if gs is None:
  gs=energy(density(np.array([occ/2,occ/2])));gs.update(request_hash=ground_key,cached_scf_energy_hartree=float(scf.chkfile.load(str(chk_path),'scf/e_tot')))
  gs['difference_from_cached_scf_hartree']=gs['energy_hartree']-gs['cached_scf_energy_hartree']
  if abs(gs['difference_from_cached_scf_hartree'])>CONFIG['ground_energy_check_hartree']:raise ValueError('UKS frozen ground differs from cached ground SCF')
  write(gs_path,gs)
 records=[]
 for target in case['targets']:
  data=dict(id=target['id'],tbe_ev=target['tbe_ev'])
  try:
   if 'checkpoint' not in target:raise ValueError(target['prior_exclusion'])
   r,coords=state_from(target);h=target['hole'];p=target['particle']
   fingerprint=spectrum_fingerprint(r.reference_state.coefficients,r.reference_state.overlap,r.reference_state.alpha_occupations+r.reference_state.beta_occupations,r.reference_state.orbital_energies_hartree)
   assert fingerprint==ref_fingerprint and np.array_equal(coords,xyz)
   mapping.validate_orbitals(np.array([h,p]));dh=int(mapping.reference_indices[h]);dp=int(mapping.reference_indices[p]);assert occ[dh]==2 and occ[dp]==0
   data.update(hole=h,particle=p,dft_hole=dh,dft_particle=dp,hole_overlap=float(mapping.overlaps[h]),particle_overlap=float(mapping.overlaps[p]),checkpoint_sha256=digest(target['checkpoint']))
   se_key=key(dict(checkpoint_sha256=data['checkpoint_sha256'],source_hashes=source_hashes));sepath=out/f'{target["id"]}_se.json';se=cached(sepath,se_key)
   if se is None:
    mixed=np.array([(s.coefficients*o)@s.coefficients.T for o in r.occupations]);triplet=np.array([(s.coefficients*o)@s.coefficients.T for o in r.triplet_occupations]);assert np.max(abs(mixed.sum(axis=0)-triplet.sum(axis=0)))<1e-12
    _,em=ctx.fock_and_energy(mixed);_,et=ctx.fock_and_energy(triplet);ef=2*em-et-r.ground_energy_hartree
    se=dict(request_hash=se_key,ground_hartree=r.ground_energy_hartree,mixed_hartree=em,triplet_hartree=et,frozen_excitation_hartree=ef,relaxed_excitation_hartree=r.excitation_energy_hartree,relaxation_hartree=r.excitation_energy_hartree-ef)
    assert abs(se['frozen_excitation_hartree']+se['relaxation_hartree']-se['relaxed_excitation_hartree'])<1e-12
    write(sepath,se)
   mixed_occ=np.array([occ/2,occ/2]);mixed_occ[0,dh]=0;mixed_occ[0,dp]=1
   triplet_occ=np.array([np.maximum(*mixed_occ),np.minimum(*mixed_occ)])
   dm=density(mixed_occ);dt=density(triplet_occ)
   assert np.max(abs(dm.sum(axis=0)-dt.sum(axis=0)))<1e-12
   req=key(dict(**parameters,hole=dh,particle=dp));dpath=out/f'{target["id"]}_dft.json';df=cached(dpath,req)
   if df is None:
    em=energy(dm);et=energy(dt);exc=2*em['energy_hartree']-et['energy_hartree']-gs['energy_hartree']
    df=dict(request_hash=req,ground_hartree=gs['energy_hartree'],mixed=em,triplet=et,frozen_excitation_hartree=exc)
    if name in ['nitroxyl','uracil'] and target==case['targets'][0]:
     flip=energy(dm[::-1].copy());df['spin_flip_error_hartree']=flip['energy_hartree']-em['energy_hartree'];assert abs(df['spin_flip_error_hartree'])<1e-8
    write(dpath,df)
   hybrid=df['frozen_excitation_hartree']+se['relaxation_hartree']
   data.update(qualified=True,se_roks_ev=se['relaxed_excitation_hartree']*EV,se_roks_error_ev=se['relaxed_excitation_hartree']*EV-target['tbe_ev'],se_frozen_ev=se['frozen_excitation_hartree']*EV,se_relaxation_ev=se['relaxation_hartree']*EV,dft_frozen_ev=df['frozen_excitation_hartree']*EV,hybrid_ev=hybrid*EV,hybrid_error_ev=hybrid*EV-target['tbe_ev'],dft_frozen_error_ev=df['frozen_excitation_hartree']*EV-target['tbe_ev'])
  except Exception as exc:data.update(qualified=False,exclusion_reason=str(exc),traceback=traceback.format_exc())
  records.append(data);write(out/'result.json',dict(host=HOST,cohort=case['cohort'],reference_sha256=refhash,backend_checkpoint_sha256=digest(chk_path),parameters=parameters,ground=gs,records=records));print(name,data,flush=True)
 return records

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--molecule');args=parser.parse_args();cases=plan();write(OUT/'plan.json',dict(host=HOST,cases=cases))
 for case in cases:
  if args.molecule and args.molecule!=case['name']:continue
  try:run(case)
  except Exception as exc:
   write(OUT/case['name']/'failure.json',dict(host=HOST,error=str(exc),traceback=traceback.format_exc()));print('CASE FAILURE',case['name'],traceback.format_exc(),flush=True)
