"""Development-only: preserve native on-site physics, add missing nonlocal response.

Two fixed parameter-free masks. Selection uses ONLY the original ten development
states, not either new cohort. No lambda or energy-offset tuning is performed.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
import sys,json,numpy as np
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,state_from,write,digest
from matrix_gfn2 import GFN2Calculator,FrozenResponseReference,save_response_reference
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
import evaluate
from pyscf import lib
lib.num_threads(4)
PREV=Path('/home/enzo/MOUSE-roks-repair-20261030/results')
OLD=Path('/home/enzo/MOUSE-deltascf-validation-20261029/hybrid_spectrum_20261029/frozen_relaxation')
LABELS=['mixed_nonlocal_spin','mixed_nonlocal_charge_spin'];evaluate.LABELS=LABELS
protocol=dict(scope='original ten development states only',lambda_exchange=.9,energy_offset=0.,
 spin='native GFN2 on-site shell blocks; DFT-derived interatomic shell blocks',
 charge_variants=['zero additional charge response','DFT-minus-native charge correction on interatomic shell blocks only'],
 parameters_fitted_to_excited_energies=0,selection='minimum npi MAE among profiles with total MAE no worse than existing shell-response model; original development states only',
 new_cohorts_used_for_selection=False)
p=BASE/'nonlocal_screen_protocol.json'
if p.exists():assert json.loads(p.read_text())==protocol
else:write(p,protocol)
for original in plan():
 name=original['name'];prior=json.loads((OLD/name/'result.json').read_text())['records']
 targets=[t for t in original['targets'] if any(r['id']==t['id'] and r['qualified'] for r in prior)]
 if not targets:continue
 case=dict(original,targets=targets);r,xyz=state_from(targets[0]);s=r.reference_state
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
 ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r.ground_energy_hartree))
 with np.load(PREV/'response_kernel'/name/'kernel.npz') as f:
  wd=f['spin_coupling'];uc=f['charge_correction'];pop=f['reference_shell_populations'];shell=f['orbital_shell']
 atoms=np.asarray(ctx.kernel.partition.shell_atom);mask=atoms[:,None]!=atoms[None,:]
 w=np.where(mask,wd,ctx.polarization.coupling_hartree)
 out=BASE/'results/models'/name;out.mkdir(exist_ok=True)
 for label,u in [('mixed_nonlocal_spin',np.zeros_like(uc)),('mixed_nonlocal_charge_spin',np.where(mask,uc,0.))]:
  reference=FrozenResponseReference(s.atomic_numbers,shell,w,'native-onsite/PBE0-nonlocal/'+digest(PREV/'response_kernel'/name/'kernel.npz')+'/'+label,.9,u,pop)
  file=out/(label+'.npz')
  if not file.exists():save_response_reference(file,reference)
 evaluate.run(case)
