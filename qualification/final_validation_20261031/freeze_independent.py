"""Freeze a small new holdout; never print numerical reference labels."""
from pathlib import Path
import socket,json,hashlib,os,re,datetime,subprocess
assert socket.getfqdn()=='oracle.sns.it'
P=Path('/home/enzo/MOUSE-publication-validation-20261030');W=Path('/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration');OUT=W/'final_validation';Q=P/'reference/QUESTDB'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
norm=lambda s:re.sub('[^a-z0-9]','',s.lower())
protocol=json.loads((OUT/'protocol.json').read_text());dest=OUT/'independent';dest.mkdir(exist_ok=True)
if (dest/'manifest.json').exists():raise SystemExit('Frozen manifest retained')
previous=json.loads((P/'blind/manifest.json').read_text());old={r['molecule'] for r in previous['states']};exposed=json.loads((P/'blind/exposure_audit.json').read_text())['excluded']
names={r[0] for r in protocol['independent']['selection']};evidence={n:[] for n in names}
# Conservative path audit, excluding library/reference/source trees and this new campaign.
skip={'source','.git','.venv','node_modules','__pycache__','reference','datasets','envs','deps','final_validation','software'}
for root in Path('/home/enzo').glob('MOUSE-*'):
 if not root.is_dir():continue
 for here,dirs,files in os.walk(root):
  dirs[:]=[x for x in dirs if x not in skip]
  for n in names:
   if norm(n) in norm(here) and len(evidence[n])<8:evidence[n].append(here)
assert not any(n in old or n in exposed or evidence[n] for n in names), 'Prior prediction/preparation exposure detected: '+str(evidence)
xyz={norm(p.stem):p for p in (Q/'geometries/xyz').rglob('*.xyz')};meta={p.stem:p for p in (Q/'data/json').rglob('*.json')};rows=[];labels={}
for name,family,ordinal in protocol['independent']['selection']:
 records=json.loads(meta[name].read_text());eligible=[]
 for idx,r in enumerate(records):
  fam='CT' if r.get('Special ?') in ['sCT','wCT'] else 'Rydberg' if r.get('V/R')=='R' else 'n_pi' if r.get('V/R')=='V' and r.get('Type')=='npi' else 'pi_pi' if r.get('V/R')=='V' and r.get('Type')=='ppi' else None
  if r.get('Spin')==1 and fam==family and 'TBE/AVTZ' in r:eligible.append((idx,r))
 idx,r=eligible[ordinal-1];sid=f'I{len(rows)+1:03d}';geometry=xyz[norm(name)];data=geometry.read_bytes();out=dest/'geometries'/f'{sid}.xyz';out.parent.mkdir(exist_ok=True);out.write_bytes(data)
 l=data.decode().splitlines();nat=int(l[0]);elements=[x.split()[0] for x in l[2:2+nat]];assert set(elements)<={'H','C','N','O'}
 state=r['State'];same=[rr for rr in records[:idx+1] if rr.get('Spin')==1 and rr.get('State')==state]
 rows.append(dict(id=sid,molecule=name,family=family,state_label=state,ordinal_same_symmetry=len(same),n_atoms=nat,geometry_sha256=sha(out),metadata_source=str(meta[name].relative_to(Q)),metadata_index=idx,partial_double_character=r.get('Special ?')=='PD'))
 labels[sid]=dict(tbe_ev=r['TBE/AVTZ'],method=r.get('Method'),safe=r.get('Safe ? (~50 meV)'))
f=dest/'sealed_labels.json';f.write_text(json.dumps(labels,indent=2)+'\n');f.chmod(0o400)
manifest=dict(schema='mouse.independent.six.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(OUT/'protocol.json'),reference_commit=subprocess.check_output(['git','-C',str(Q),'rev-parse','HEAD'],text=True).strip(),exposure_audit=evidence,old_audit_sha256=sha(P/'blind/exposure_audit.json'),sealed_labels_sha256=sha(f),candidate_frozen_before_predictions=True,states=rows,CT_shortfall=True)
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print([(r['id'],r['molecule'],r['family'],r['state_label']) for r in rows])
