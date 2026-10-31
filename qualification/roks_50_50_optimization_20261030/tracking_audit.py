"""Read final accepted/anchor bundles; no new electronic optimization."""
import sys,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
BASE=Path('/home/enzo/MOUSE-roks-50-50-optimization-20261030')
spec=importlib.util.spec_from_file_location('campaign',BASE/'source/qualification/run_campaign_recovered.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
from matrix_link.scan import QMScanBackend
from matrix_link.composite_roks_component import restore_accepted
from matrix_link.qmqm_component import CappedQMComponent
from matrix_apoc import singlet_csf_overlap
rows=[]
with c.threadpool_limits(limits=2):
 for name in ['nitroxyl','acetaldehyde','maleimide']:
  p=c.prepare(name)
  backend=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks_50_50',deltascf_promotion=tuple(p['promotion']),native_roks_composite_response_paths=tuple(p['response_paths']),native_roks_composite_saddle_indices=tuple(p['saddle_indices']))
  for branch in ['main','negative_mode_minus','negative_mode_plus']:
   root=BASE/'results'/name/branch/'roks_50_50'
   if not (root/'roks_50_50_accepted/accepted.json').exists():continue
   component=SimpleNamespace(backend=backend,run_dir=root,reference=None,anchor=None);restore_accepted(component)
   scores=[]
   for a,r,ap,rp in zip(component.anchor[0].components,component.reference[0].components,component.anchor[1],component.reference[1]):
    cross=CappedQMComponent._cross_overlap(a.reference_state,r.reference_state);scores.append(float(singlet_csf_overlap(ap,rp,cross)[0]))
   rows.append(dict(molecule=name,branch=branch,anchor_overlaps=scores,maximum_gradient=float(c.np.max(abs(component.reference[0].gradient_hartree_per_bohr)))))
record=dict(host=c.HOST,rows=rows);c.write(BASE/'results/tracking_audit.json',record)
for r in rows:print(r)
