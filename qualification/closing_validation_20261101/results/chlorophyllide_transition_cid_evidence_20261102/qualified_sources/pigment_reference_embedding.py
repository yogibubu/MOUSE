"""Checkpointed native-valence embedding of completed pigment DFT references.

No SCF is rerun, no excitation labels are read, and no excitation energy is
claimed. The complete converged MO projector also handles the linear
redundancies removed in the large augmented Gaussian basis.
"""
from pathlib import Path
import socket, subprocess, argparse, json, hashlib, time, os
assert socket.getfqdn() == 'mac-2.homenet.telecomitalia.it'
assert subprocess.check_output(['scutil','--get','ComputerName'], text=True).strip() == 'Mac Studio di Vincenzo'
from gaussian_embedding_basis import source_molecule
from prepare_fresh import HERE
from matrix_gfn2.electronic_state import GFN2OrbitalState, GFN2OrbitalBasis
from dataclasses import fields
import numpy as np
from scipy.linalg import eigh
from pyscf import gto
from pyscf.data.elements import ELEMENTS
from threadpoolctl import threadpool_limits

def sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, data):
 temporary=path.with_suffix(path.suffix+'.tmp')
 temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)

def power(matrix, exponent):
 w,u=eigh(matrix)
 assert w.min()>1e-10, ('Unrepresentable native embedding',float(w.min()))
 return (u*w**exponent)@u.T

def run(name):
 out=HERE/'pigment_embedding'/name;out.mkdir(parents=True,exist_ok=True)
 ground=HERE/'pigment_preflight'/name/'native_ground.npz'
 paths={label:HERE/'pigment_dft_reference'/name/label/'reference.npz' for label in ['PBE0','CAM']}
 expected={'native_ground':sha(ground),**{label:sha(path) for label,path in paths.items()}}
 cert=out/'certificate.json'
 if cert.exists():
  previous=json.loads(cert.read_text());assert previous['input_sha256']==expected
  assert sha(out/'embedding.npz')==previous['output_sha256']
  return previous
 with np.load(ground) as z:
  basis=GFN2OrbitalBasis(**{f.name:z['basis_'+f.name].copy() for f in fields(GFN2OrbitalBasis)})
  kw={f.name:z[f.name].copy() for f in fields(GFN2OrbitalState) if f.name!='basis'}
 kw['model_id']=str(kw['model_id']);kw['ao_shell']=tuple(kw['ao_shell'].tolist())
 for field in ['electron_count','charge','multiplicity']:kw[field]=int(kw[field])
 kw['total_energy_hartree']=float(kw['total_energy_hartree']);state=GFN2OrbitalState(**kw,basis=basis)
 references={}
 for label,path in paths.items():
  with np.load(path) as z:references[label]={key:z[key].copy() for key in z.files}
 atoms=[(ELEMENTS[int(z)],r) for z,r in zip(state.atomic_numbers,state.coordinates_bohr)]
 mol=gto.M(atom=atoms,unit='Bohr',basis='aug-cc-pVDZ',spin=0,charge=state.charge,verbose=0)
 small,transform,basis_certificate=source_molecule(state,return_certificate=True)
 cross=gto.intor_cross('int1e_ovlp',mol,small)@transform
 ref=references['PBE0'];sd=ref['overlap'];cd=ref['coefficients'];no=state.electron_count//2
 nc=int(np.count_nonzero(ref['occupations']==2)-no);assert nc>=0
 cval=cd[:,nc:];x=cval@(cval.T@cross)
 gram=x.T@sd@x;lift=x@power(gram,-.5)@power(state.overlap,.5)
 error=float(np.max(abs(lift.T@sd@lift-state.overlap)))
 core_error=float(np.max(abs(cd[:,:nc].T@sd@lift))) if nc else 0.
 assert error<1e-8 and core_error<1e-8
 arrays=dict(embedding=lift,overlap=state.overlap,native_coefficients=state.coefficients)
 diagnostics={}
 for label,r in references.items():
  assert np.max(abs(r['coordinates_bohr']-state.coordinates_bohr))<1e-7
  assert np.array_equal(r['atomic_numbers'],state.atomic_numbers)
  p=r['coefficients'].T@sd@lift
  fock=(p.T*r['orbital_energies'])@p
  spectral=p@state.coefficients
  represented=np.sum(spectral**2,axis=0)
  assert np.max(abs(represented-1))<1e-7
  arrays[label+'_projected_fock']=fock;arrays[label+'_native_MO_spectral_amplitudes']=spectral
  arrays[label+'_orbital_energies']=r['orbital_energies'];arrays[label+'_occupations']=r['occupations']
  diagnostics[label]=dict(projected_fock_asymmetry=float(np.max(abs(fock-fock.T))),spectral_norm_max_error=float(np.max(abs(represented-1))),full_retained_MOs=r['coefficients'].shape[1])
 np.savez_compressed(out/'embedding.npz',**arrays)
 report=dict(molecule=name,host=socket.getfqdn(),input_sha256=expected,output_sha256=sha(out/'embedding.npz'),native_orbitals=len(state.overlap),frozen_core_orbitals=nc,embedding_overlap_max_error=error,core_overlap_max_error=core_error,basis_reconstruction=basis_certificate,references=diagnostics,new_SCC_calls=0,new_DFT_SCF_calls=0,excitation_labels_read=False,status='Native embedding and projected reference operators complete; OO response and state calculations still required')
 write(cert,report);return report

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('name');parser.add_argument('--wait-for-references',action='store_true');args=parser.parse_args()
 out=HERE/'pigment_embedding'/args.name;out.mkdir(parents=True,exist_ok=True)
 lock=out/'worker.lock'
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  while not all((HERE/'pigment_dft_reference'/args.name/label/'reference.npz').exists() for label in ['PBE0','CAM']):
   if not args.wait_for_references:raise RuntimeError('References incomplete; saved checkpoints preserved')
   write(out/'status.json',dict(pid=os.getpid(),status='Waiting for converged DFT references; no duplicate SCF'))
   time.sleep(20)
  write(out/'status.json',dict(pid=os.getpid(),status='Constructing compact native embedding'))
  with threadpool_limits(2):report=run(args.name)
  write(out/'status.json',dict(pid=os.getpid(),status='Complete',certificate='certificate.json'))
  print(json.dumps(report,indent=2),flush=True)
 finally:lock.unlink(missing_ok=True)
