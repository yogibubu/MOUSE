"""Audit saved pigment CI windows without repeating any electronic work."""
from pathlib import Path
import socket, json, argparse, time, subprocess
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-closing-validation-20261101')
p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--reuse-analysis',action='store_true');p.add_argument('--screening-frontier',type=int,default=8);p.add_argument('--frontiers',nargs='+',type=int,default=[6,7,8]);a=p.parse_args()
folders=[HERE/f'pigment_cid_expansion/Beta_carotene/screening_{a.screening_frontier}/frontier_{n}' for n in a.frontiers]
while not all((f/'result.json').exists() for f in folders):
 if not a.wait:raise RuntimeError('Convergence sequence incomplete')
 time.sleep(20)
rows=[]
for n,folder in zip(a.frontiers,folders):
 if not a.reuse_analysis or not (folder/'analysis.json').exists():
  subprocess.run(['/Users/vincenzobarone/.venvs/matrix/bin/python',str(HERE/'analyze_pigment_expansion.py'),'Beta_carotene',str(n),'--screening-frontier',str(a.screening_frontier)],check=True)
 r=json.loads((folder/'result.json').read_text());analysis=json.loads((folder/'analysis.json').read_text());states=analysis['states'][:3]
 row=dict(frontier=n,screening_frontier=a.screening_frontier,basis_size=r['basis_size'],states=states,root_tracking=analysis['root_tracking'],principal_submatrix=analysis['baseline_principal_submatrix'],current_process_wall_seconds=r['current_process_wall_seconds'],matrix_checkpoint_reused=r['matrix_checkpoint_reused'])
 if rows:row['energy_increments_ev']=[s['excitation_ev']-old['excitation_ev'] for s,old in zip(states,rows[-1]['states'])]
 rows.append(row)
report=dict(host=socket.getfqdn(),molecule='Beta_carotene',rows=rows,parameters_retuned=False,weights=[.5,.5],lambda_exchange=.9,eta=.6,offset_ev=0,source_geometry_changed=False,scope='Post-exposure fixed-screening CI convergence diagnostic; vertical gaps, not solution maxima or adiabatic origins',all_windows_weak_S1_below_intense_S2=all(r['states'][0]['oscillator_strength']<r['states'][1]['oscillator_strength']/50 for r in rows))
(HERE/('beta_carotene_convergence.json' if a.screening_frontier==8 else f'beta_carotene_convergence_w{a.screening_frontier}.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
