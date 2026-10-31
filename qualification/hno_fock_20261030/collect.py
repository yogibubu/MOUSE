"""Remote checkpoint/provenance collection, without electronic recalculation."""
import importlib.util,json,hashlib,socket
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
BASE=Path('/home/enzo/MOUSE-hno-fock-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from matrix_link.composite_roks_component import restore_accepted
from matrix_gfn2.roks_tracking import active_orbital_character
from matrix_gfn2 import load_response_reference
rows=[]
for stage in ['resume','spectrum_resume','ground_resume','newton_resume']:
 root=BASE/'results/nitroxyl'/stage
 if not (root/'roks_50_50/roks_50_50_accepted/accepted.json').exists():continue
 b=p.b if stage in ['resume','spectrum_resume'] else replace(p.b,native_roks_composite_ground_recovery='accepted_retry')
 c=SimpleNamespace(backend=b,run_dir=root/'roks_50_50',reference=None,anchor=None);restore_accepted(c)
 r=c.reference[0];s=r.components[0].reference_state;np=p.np;n=int(np.sum(s.alpha_occupations))
 row=dict(stage=stage,energy_hartree=r.energy_hartree,maximum_gradient=float(np.max(abs(r.gradient_hartree_per_bohr))),
   indices=list(r.orbital_saddle_indices),component_csf_overlap=r.component_csf_overlap,native_ground_gap_hartree=float(s.orbital_energies_hartree[n]-s.orbital_energies_hartree[n-1]),
   coordinates_angstrom=(s.coordinates_bohr/p.ANGSTROM_TO_BOHR).tolist(),active_character=[active_orbital_character(a,z,load_response_reference(path).anchor_coordinates_bohr) for a,z,path in zip(c.anchor[0].components,r.components,b.native_roks_composite_response_paths)])
 row['complete']=json.loads((root/'complete.json').read_text()) if (root/'complete.json').exists() else None
 rows.append(row)
summary=dict(host=socket.getfqdn(),rows=rows,certified_minima=0,all_jobs_complete=True,
 final_production_changes=['per-call temperature homotopy control; fixed-temperature Newton/accepted-guess certification','ordered S0 energy guesses','correct generalized eigenproblem on unconverged direct SCC exit','no unrelated S0 scaling-observable SCF at ROKS optimizer completion'],
 numerical_experiments_not_retained=['damped nonlinear Newton line search','inner Newton iterations before native certification'],
 derivative_check=json.loads((BASE/'derivative_check.json').read_text()),
 strict_gates=dict(fock_hartree=1e-10,canonical_eigen_hartree=1e-12,maximum_force_hartree_per_bohr=3e-5),weights=[.5,.5],lambda_exchange=.9,energy_offset=0.)
(BASE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
# Preserve the baseline JSON lines before the fixed decomposition replaced its
# matrix files. Original baseline stdout remains an independent artifact.
baseline=[json.loads(line) for line in (BASE/'probe.log').read_text().splitlines() if line.startswith('{')]
(BASE/'decomposition_baseline.json').write_text(json.dumps(dict(rows=baseline),indent=2)+'\n')
print(json.dumps(dict(stages=len(rows),final_gradient=rows[-1]['maximum_gradient'],certified_minima=0)))
