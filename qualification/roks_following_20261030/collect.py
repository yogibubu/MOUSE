"""Collect retained accepted endpoints, not trial states; remote host only."""
import importlib.util,json,hashlib
from pathlib import Path
from types import SimpleNamespace
BASE=Path('/home/enzo/MOUSE-roks-following-20261030')
spec=importlib.util.spec_from_file_location('campaign',BASE/'source/qualification/run_campaign.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
from matrix_link.composite_roks_component import restore_accepted,_models
from matrix_gfn2.roks_tracking import active_orbital_character
from dataclasses import replace
rows=[]
with c.threadpool_limits(limits=2):
 for name in ['nitroxyl','acetaldehyde','maleimide']:
  p=c.prepare(name)
  for branch in ['resume','ground_recovery']:
   folder=BASE/'results'/name/branch
   if not (folder/'roks_50_50/roks_50_50_accepted/accepted.json').exists():continue
   b=c.QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks_50_50',deltascf_promotion=tuple(p['promotion']),
       native_roks_composite_response_paths=tuple(p['response_paths']),native_roks_composite_saddle_indices=tuple(p['saddle_indices']),
       native_roks_composite_following='character_adaptive_index' if name=='nitroxyl' else 'character',
       native_roks_composite_ground_recovery='accepted_retry' if branch=='ground_recovery' else None,processors=2,properties=('energy','gradient'))
   comp=SimpleNamespace(backend=b,run_dir=folder/'roks_50_50',reference=None,anchor=None);restore_accepted(comp)
   r=comp.reference[0];xyz=r.components[0].reference_state.coordinates_bohr/c.ANGSTROM_TO_BOHR
   modes=[]
   for audit in ['terminal_audit','terminal_recovery_audit']:
    f=BASE/'results'/name/(audit+'.json')
    if f.exists() and ((audit=='terminal_audit')==(branch=='resume')):modes.append(json.loads(f.read_text()))
   row=dict(molecule=name,branch=branch,energy_hartree=r.energy_hartree,maximum_gradient=float(c.np.max(abs(r.gradient_hartree_per_bohr))),
       coordinates_angstrom=xyz.tolist(),component_csf_overlap=r.component_csf_overlap,orbital_indices=list(r.orbital_saddle_indices),
       active_character=[active_orbital_character(a,z,ref.anchor_coordinates_bohr) for a,z,ref in zip(comp.anchor[0].components,r.components,_models(b))],
       terminal_audit=modes,complete=json.loads((folder/'complete.json').read_text()) if (folder/'complete.json').exists() else None)
   if name=='acetaldehyde':
    row['geometry']=dict(co_angstrom=float(c.np.linalg.norm(xyz[0]-xyz[2])),cc_angstrom=float(c.np.linalg.norm(xyz[0]-xyz[1])),aldehydic_ch_angstrom=float(c.np.linalg.norm(xyz[0]-xyz[3])))
   rows.append(row)
 output=dict(host=c.HOST,weights=[.5,.5],lambda_exchange=.9,offset_hartree=0.,rows=rows,
     all_jobs_complete=all(row['complete'] is not None or (BASE/'results'/row['molecule']/('failure.json' if row['branch']=='resume' else 'ground_recovery_failure.json')).exists() for row in rows),
     certified_minima=sum(bool(row['complete'] and row['complete'].get('passed')) for row in rows))
 c.write(BASE/'summary.json',output)
 print(json.dumps(output,indent=2))
