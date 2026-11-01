"""Audit fixed-Hamiltonian and coherent active-exclusion/CI convergence."""
from pathlib import Path
import socket,json,hashlib,argparse
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--frontiers',type=int,nargs='+',default=[11,12,13]);a=p.parse_args();assert all(y==x+1 for x,y in zip(a.frontiers,a.frontiers[1:]))
protocol=json.loads((HERE/'beta_convergence_protocol_v2.json').read_text());root=HERE/'pigment_cid_expansion/Beta_carotene'
def load(n,w):
 r=json.loads((root/f'screening_{w}/frontier_{n}/result.json').read_text())
 assert r['weights']==[.5,.5] and r['lambda_exchange']==.9 and r['bath_efficiency_eta']==.6 and r['energy_offset_ev']==0
 assert 'observable_phase_convention' in r,'Observable refresh not completed'
 return r
fixed=[load(n,16) for n in a.frontiers];coherent=[load(n,n) for n in a.frontiers]
assert len({r['model_sha256'] for r in fixed+coherent})==1
for n in a.frontiers:
 cert=json.loads((HERE/f'pigment_screening_expansion/Beta_carotene/frontier_{n}/certificate.json').read_text());assert cert['active_frontier']==n and cert['positive_dielectric_by_construction']
def increments(rows):return [[y['states'][i]['excitation_ev']-x['states'][i]['excitation_ev'] for i in range(3)] for x,y in zip(rows,rows[1:])]
def passes(rows):
 inc=increments(rows)[-2:];return len(inc)==2 and all(abs(v)<=protocol['CI_energy_tolerance_ev'] for row in inc for v in row)
result=dict(protocol=protocol,frontiers=a.frontiers,fixed_screening_window=16,fixed_Hamiltonian_increments_ev=increments(fixed),coherent_W_equals_N_increments_ev=increments(coherent),fixed_Hamiltonian_converged=passes(fixed),coherent_model_converged=passes(coherent),converged=passes(fixed) and passes(coherent),coherent_states=coherent[-1]['states'][:3],fixed_screening_states=fixed[-1]['states'][:3],same_geometry_and_parameters=True,coherent_result_sha256=hashlib.sha256((root/f'screening_{a.frontiers[-1]}/frontier_{a.frontiers[-1]}/result.json').read_bytes()).hexdigest(),original_W_sensitivity_criterion='Superseded for the stated mathematical reason; failed sensitivity checks retained separately, not relabelled as converged',scope='Numerical convergence of the explicit rank-two prepared model on the source geometry; no general accuracy or pigment Hessian qualification')
(HERE/'beta_convergence_assessment.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
