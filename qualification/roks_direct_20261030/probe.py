"""Validate the absolute GFN2 functional before removing the S0 SCC dependency."""
import os,sys,json,socket
from pathlib import Path
from types import SimpleNamespace
BASE=Path('/home/enzo/MOUSE-roks-direct-20261030');OLD=Path('/home/enzo/MOUSE-roks-following-20261030')
assert socket.getfqdn()=='oracle.sns.it';os.environ['MOUSE_VALIDATION_ROOT']=str(BASE)
sys.path.insert(0,'/home/enzo/MOUSE-roks-independent-20261030');import remote_bootstrap
for path in (BASE/'source/packages').glob('*/src'):sys.path.insert(0,str(path))
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator,load_response_reference
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.roks import evaluate_gfn2_roks_singlet
from matrix_link.scan import QMScanBackend
from matrix_link.composite_roks_component import restore_accepted

def retained(name):
 root=OLD/'results'/name;p=json.loads((root/'prepared.json').read_text());branch='resume' if name=='nitroxyl' else 'ground_recovery'
 backend=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks_50_50',deltascf_promotion=tuple(p['promotion']),native_roks_composite_response_paths=tuple(p['response_paths']),native_roks_composite_saddle_indices=tuple(p['saddle_indices']),native_roks_composite_following='character_adaptive_index' if name=='nitroxyl' else 'character',native_roks_composite_ground_recovery=None if name=='nitroxyl' else 'accepted_retry')
 c=SimpleNamespace(backend=backend,run_dir=root/branch/'roks_50_50',reference=None,anchor=None);restore_accepted(c)
 return p,c

if __name__=='__main__':
 with threadpool_limits(limits=2):
  rows=[]
  for name in ['nitroxyl','acetaldehyde','maleimide']:
   p,c=retained(name);calc=GFN2Calculator(p['atoms'],temperature_kelvin=0.)
   # Any electronic SCC call makes this qualification fail immediately.
   calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('unexpected ground SCC'))
   for i,old in enumerate(c.reference[0].components):
    xyz=old.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR;ref=load_response_reference(p['response_paths'][i])
    context=prepare_gfn2_deltascf(calc,xyz,spin_reference=ref,orbital_reference=old.reference_state)
    fm,em=context.fock_and_energy(old.densities());ft,et=context.fock_and_energy(old.densities(triplet=True))
    new=evaluate_gfn2_roks_singlet(calc,xyz,*old.occupations,spin_reference=ref,reference_result=old,reference_policy='geometry_only',gradient=True,orbital_solver='gmf',saddle_index=old.orbital_saddle_index,orbital_tolerance=1e-10,maximum_iterations=150)
    row=dict(molecule=name,component=i,fock_error=float(max(np.max(abs(fm-old.fock_mixed)),np.max(abs(ft-old.fock_triplet)))),
      energy_error=float(2*em-et-old.energy_hartree),relaxed_energy_error=float(new.energy_hartree-old.energy_hartree),gradient_error=float(np.max(abs(new.gradient_hartree_per_bohr-old.gradient_hartree_per_bohr))),gap_is_none=new.excitation_energy_hartree is None)
    rows.append(row);print(json.dumps(row),flush=True)
   (BASE/'probe.json').write_text(json.dumps(dict(host=socket.getfqdn(),rows=rows),indent=2)+'\n')
