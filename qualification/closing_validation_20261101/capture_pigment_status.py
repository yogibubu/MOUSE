"""Read-only verified-host status snapshot; never restarts electronic work."""
from pathlib import Path
import json,socket,hashlib,datetime
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
P=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-closing-validation-20261101')
report=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.getfqdn(),pigments=[],accuracy_pool_member=False)
for name in ['Beta_carotene','Chlorophyllide_a']:
 folder=P/'pigment_preflight'/name;row=dict(name=name,native=json.loads((folder/'result.json').read_text()),embedding=json.loads((folder/'embedding_basis_certificate.json').read_text()),native_checkpoint_sha256=hashlib.sha256((folder/'native_ground.npz').read_bytes()).hexdigest(),references={})
 for label in ['PBE0','CAM']:
  r=P/'pigment_dft_reference'/name/label;cert=r/'certificate.json';progress=r/'progress.json'
  row['references'][label]=dict(converged_reference_exists=(r/'reference.npz').exists(),restart_checkpoint_exists=(r/'backend.chk').exists(),progress=json.loads(progress.read_text()) if progress.exists() else None,certificate=json.loads(cert.read_text()) if cert.exists() else None)
 embedding=P/'pigment_embedding'/name
 row['compact_embedding_certificate']=json.loads((embedding/'certificate.json').read_text()) if (embedding/'certificate.json').exists() else None
 row['compact_embedding_status']=json.loads((embedding/'status.json').read_text()) if (embedding/'status.json').exists() else None
 for stage,relative in [('native_stationarity','pigment_preflight/'+name+'/stationary_native_certificate.json'),('compact_anchor','pigment_compact_reference/'+name+'/certificate.json'),('full_response','pigment_full_response/'+name+'/ri_xc_chart/certificate.json'),('runtime_manifest','pigment_models/'+name+'/manifest.json'),('ROKS','pigment_roks/'+name+'/result.json'),('screening','pigment_screening/'+name+'/certificate.json'),('CID','pigment_cid/'+name+'/result.json')]:
  path=P/relative;row[stage]=json.loads(path.read_text()) if path.exists() else None
 row['ROKS_progress']=[json.loads(path.read_text()) for path in sorted((P/'pigment_roks'/name).glob('state_*/progress.json'))]
 report['pigments'].append(row)
report['sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/n for n in ['pigment_dft_reference.py','pigment_preflight.py','gaussian_embedding_basis.py','check_pigment_basis.py','pigment_reference_embedding.py']]}
(P/'pigment_status.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({r['name']:r['references'] for r in report['pigments']},indent=2))
