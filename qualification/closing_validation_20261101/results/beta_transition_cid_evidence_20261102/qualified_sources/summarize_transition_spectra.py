"""Assess a frozen coherent-transition screening protocol on any prepared case."""
from pathlib import Path
import argparse,socket,json,time,subprocess,sys
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--protocol',required=True);p.add_argument('--wait',action='store_true');a=p.parse_args();definition=json.loads((HERE/a.protocol).read_text());ns=definition['coherent_frontiers'][a.name];w=definition['fixed_screening_frontier'];prefix=definition['artifact_prefixes'][a.name];roots=definition['roots_monitored'][a.name]
def folder(n,screen):return HERE/f'pigment_cid_transition/{a.name}/screening_{screen}/frontier_{n}'
pairs=list(dict.fromkeys([(n,n) for n in ns]+[(n,w) for n in ns]))
while not all((folder(n,s)/'result.json').exists() for n,s in pairs):
 if not a.wait:raise RuntimeError('Coherent transition validation sequence incomplete')
 time.sleep(10)
def read(n,s):return json.loads((folder(n,s)/'result.json').read_text())
def changes(rows):return [[b['states'][k-1]['excitation_ev']-aa['states'][k-1]['excitation_ev'] for k in roots] for aa,b in zip(rows,rows[1:])]
coherent=[read(n,n) for n in ns];fixed=[read(n,w) for n in ns];dc=changes(coherent);df=changes(fixed);threshold=definition['energy_tolerance_ev']
protocol=dict(molecule=a.name,artifact_prefix=prefix,weights=definition['weights'],lambda_exchange=definition['lambda_exchange'],eta=definition['eta'],energy_offset_ev=definition['energy_offset_ev'],energy_tolerance_ev=threshold,successive_increments_required=definition['successive_increments_required'],roots=roots,baseline_frontiers=[2],baseline_screening_frontier=5,fixed_screening_frontier=w,screening_mode='coherent_transitions',case_modes={'N2_W5':'legacy_sectors'},extra_evidence_files=['coherent_transition_screening_protocol.json'],scope=definition['scope'])
if a.name in definition.get('baseline_matrix_certificates',{}):protocol['certified_baseline_matrix_sha256']=definition['baseline_matrix_certificates'][a.name]
(HERE/f'{prefix}_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
result=dict(protocol=protocol,host=socket.getfqdn(),windows=ns,coherent_results=[str(folder(n,n)) for n in ns],fixed_results=[str(folder(n,w)) for n in ns],coherent_increments_ev=dc,fixed_Hamiltonian_increments_ev=df,coherent_converged=len(dc)>=definition['successive_increments_required'] and all(abs(v)<threshold for row in dc for v in row),fixed_Hamiltonian_converged=len(df)>=definition['successive_increments_required'] and all(abs(v)<threshold for row in df for v in row),screening_mode='coherent_transitions',parameters_retuned=False)
(HERE/f'{prefix}_convergence.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
with (HERE/f'{prefix}_audit.log').open('w') as log:subprocess.run([sys.executable,'-u',str(HERE/'audit_pigment_cid_validation.py'),'--protocol',f'{prefix}_protocol.json'],stdout=log,stderr=subprocess.STDOUT,check=True)
with (HERE/f'{prefix}_collection.log').open('w') as log:subprocess.run([sys.executable,'-u',str(HERE/'collect_pigment_cid_validation.py'),'--protocol',f'{prefix}_protocol.json','--output',f'{prefix}_evidence_20261102'],stdout=log,stderr=subprocess.STDOUT,check=True)
