"""Assemble complete analytic Hessians and compare reconverged energies."""
from pathlib import Path
import json,socket
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-pi-pi-transfer-repair-20261101')
import numpy as np
from threadpoolctl import threadpool_limits
cases=[('independent','Criegee_Intermediate',15),('last_two','Furan',27),('last_two','uracil',36)];all_reports=[]
with threadpool_limits(2):
 for cohort,name,nr in cases:
  folder=HERE/'analytic'/cohort/name;paths=[folder/f'column_{i:03d}.npz' for i in range(nr)];assert all(p.exists() for p in paths),(name,'missing columns')
  columns=[]
  for path in paths:
   with np.load(path) as z:columns.append({k:z[k].copy() for k in z.files})
  rows=json.loads((HERE/'even_sector_5'/cohort/name/'result.json').read_text())['primary']['predictions'];fd=HERE/'finite_energy'/cohort/name;center=json.loads((fd/'center.json').read_text());directions=[np.array(v) for v in center['directions']];states={};arrays={}
  for row in rows:
   identity=row['id'];gradient=columns[0][identity+'_gradient'];h=np.column_stack([c[identity+'_hessian_column'] for c in columns]);gerror=max(float(np.max(abs(c[identity+'_gradient']-gradient))) for c in columns);asym=float(np.max(abs(h-h.T)));translation=float(np.max(abs(h@np.tile(np.eye(3),(nr//3,1)))));gtranslation=float(np.max(abs(gradient.reshape(-1,3).sum(0))));assert gerror<2e-8 and asym<2e-6 and translation<2e-6 and gtranslation<2e-8,(name,identity,gerror,asym,translation,gtranslation)
   checks=[]
   for i,direction in enumerate(directions):
    pair=[]
    for eps in [1e-3,5e-4]:
     energies=[]
     for sign in [-1,1]:
      record=json.loads((fd/f'direction_{i}_eps_{eps:.8f}_sign_{sign}.json').read_text());assert record['identity']==center['identity'];energies.append(record['energies'][identity])
     em,ep=energies;dg=(ep-em)/(2*eps);dh=(ep-2*center['energies'][identity]+em)/eps**2;pair.append(dict(epsilon_bohr=eps,gradient=dg,curvature=dh,raw_gradient_error=abs(dg-gradient@direction),raw_curvature_error=abs(dh-direction@h@direction)))
    coarse,fine=pair;ge=abs((4*fine['gradient']-coarse['gradient'])/3-gradient@direction);he=abs((4*fine['curvature']-coarse['curvature'])/3-direction@h@direction);assert ge<2e-7 and he<2e-5,(name,identity,i,ge,he)
    checks.append(dict(direction=i,character='out_of_plane' if i==0 else 'mixed_internal',steps=pair,richardson_gradient_error=float(ge),richardson_curvature_error=float(he)))
   states[identity]=dict(energy_hartree=float(columns[0][identity+'_energy']),gradient_column_consistency=gerror,raw_hessian_asymmetry=asym,translation_residual=translation,gradient_translation_residual=gtranslation,independent_energy_checks=checks)
   arrays[identity+'_gradient']=gradient;arrays[identity+'_hessian']=h
  np.savez_compressed(folder/'nuclear_response.npz',**arrays);report=dict(molecule=name,cohort=cohort,coordinates=nr,states=states,method='fully analytic anchored-model nuclear response; finite differences only for independent validation',no_DFT_runtime=True,no_SCC_runtime=True)
  (folder/'hessian_certificate.json').write_text(json.dumps(report,indent=2)+'\n');all_reports.append(report);print(name,{k:(v['raw_hessian_asymmetry'],max(x['richardson_curvature_error'] for x in v['independent_energy_checks'])) for k,v in states.items()},flush=True)
 (HERE/'molecular_derivative_summary.json').write_text(json.dumps(all_reports,indent=2)+'\n')
