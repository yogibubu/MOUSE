"""Evaluate the preregistered numerical convergence thresholds, without fitting."""
from pathlib import Path
import json,argparse,socket
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--window',type=int,default=12);p.add_argument('--frontiers',type=int,nargs='+',default=[10,11,12]);p.add_argument('--screening-control',type=int,default=10);p.add_argument('--previous-window',type=int,default=10);a=p.parse_args()
protocol=json.loads((HERE/'beta_convergence_protocol.json').read_text())
def load(n,w):return json.loads((HERE/f'pigment_cid_expansion/Beta_carotene/screening_{w}/frontier_{n}/result.json').read_text())
rows=[load(n,a.window) for n in a.frontiers]
for r in rows:
 assert r['weights']==protocol['weights'] and r['lambda_exchange']==protocol['lambda_exchange'] and r['bath_efficiency_eta']==protocol['eta'] and r['energy_offset_ev']==0
assert len({r['model_sha256'] for r in rows})==1
increments=[[new['states'][i]['excitation_ev']-old['states'][i]['excitation_ev'] for i in range(3)] for old,new in zip(rows,rows[1:])]
last=increments[-protocol['CI_successive_increments_required']:]
ci_pass=len(last)==protocol['CI_successive_increments_required'] and all(abs(x)<=protocol['CI_energy_tolerance_ev'] for row in last for x in row)
old=load(a.screening_control,a.previous_window);new=load(a.screening_control,a.window)
screen=[new['states'][i]['excitation_ev']-old['states'][i]['excitation_ev'] for i in range(3)]
screen_pass=all(abs(x)<=protocol['screening_energy_tolerance_ev'] for x in screen)
result=dict(protocol=protocol,frontiers=a.frontiers,screening_window=a.window,CI_increments_ev=increments,CI_converged=ci_pass,screening_control_frontier=a.screening_control,screening_increments_ev=screen,screening_converged=screen_pass,converged=ci_pass and screen_pass,lowest_states=rows[-1]['states'][:3],next_action='Numerical criterion met' if ci_pass and screen_pass else 'Continue expansion; do not claim convergence',parameters_retuned=False)
(HERE/'beta_convergence_assessment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
