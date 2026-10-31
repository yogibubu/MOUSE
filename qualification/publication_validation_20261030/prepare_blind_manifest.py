"""Freeze an exposure-audited holdout without printing or exposing TBE labels."""
from pathlib import Path
import os,json,re,socket,hashlib,random,subprocess,datetime
assert socket.getfqdn()=='oracle.sns.it'
P=Path(__file__).resolve().parent;Q=P/'reference/QUESTDB';OUT=P/'blind';OUT.mkdir(exist_ok=True)
def norm(v):return re.sub('[^a-z0-9]','',v.lower())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
if (OUT/'manifest.json').exists():raise SystemExit('Blind manifest already frozen; preserve and resume it')
paths=[]
for p in Path('/home/enzo').glob('MOUSE-*'):
 if p!=P and p.is_dir():paths.append(p)
paths += list(Path('/home/enzo/MATRIX-runs').glob('*/docs/reports'))
paths += list(Path('/home/enzo/MATRIX-runs').glob('*/reports'))
skip={'source','.git','.venv','venv','node_modules','__pycache__','build','deps','envs','software','reference'}
existing=[norm(s) for s in (P/'inputs/local_existing_paths.txt').read_text().splitlines()]
for root in paths:
 for here,dirs,files in os.walk(root):
  dirs[:]=[s for s in dirs if s not in skip]
  existing.append(norm(here))
  existing += [norm(str(Path(here)/s)) for s in files if s.endswith(('.json','.npz','.xyz','.log','.out'))]
files=sorted((Q/'data/json').rglob('*.json'));molecules=sorted(set(f.stem for f in files));exposed={}
for name in molecules:
 key=norm(name);ev=[s for s in existing if key in s]
 if ev:exposed[name]=dict(count=len(ev),first_evidence=ev[:3])
write(OUT/'exposure_audit.json',dict(host=socket.getfqdn(),roots=[str(p) for p in paths],local_paths_sha256=sha(P/'inputs/local_existing_paths.txt'),comparison='conservative normalized name/path substring; can overexclude, cannot establish absence outside audited trees',excluded=exposed))
xyzs={norm(f.stem):f for f in (Q/'geometries/xyz').rglob('*.xyz')};candidates={k:[] for k in ['n_pi','pi_pi','CT','Rydberg']};skips={}
charged={'Pyridinium','Phenolate','Streptocyanine-C1','Streptocyanine-C3','Streptocyanine-C5'};seen=set()
for f in files:
 name=f.stem
 if name in exposed or name in charged or '+' in name:continue
 geom=xyzs.get(norm(name))
 if geom is None:skips[name]='no unique XYZ match';continue
 try:
  lines=geom.read_text().splitlines();nat=int(lines[0]);atoms=[l.split()[0] for l in lines[2:2+nat]]
  if len(atoms)!=nat or not set(atoms)<={'H','C','N','O'}:continue
 except Exception:continue
 for idx,r in enumerate(json.loads(f.read_text())):
  if r.get('Spin')!=1:continue
  kind=str(r.get('Type','')).strip();special=str(r.get('Special ?',''));vr=r.get('V/R')
  family='CT' if special in ['sCT','wCT'] else 'Rydberg' if vr=='R' else 'n_pi' if vr=='V' and kind=='npi' else 'pi_pi' if vr=='V' and kind=='ppi' else None
  if family is None or 'TBE/AVTZ' not in r:continue
  key=(name,r['State'],family,vr)
  if key in seen:continue
  seen.add(key)
  public=dict(molecule=name,state_label=r['State'],family=family,source_metadata=str(f.relative_to(Q)),metadata_index=idx,geometry_source=str(geom.relative_to(Q)),geometry_sha256=sha(geom),n_atoms=nat,n_heavy=sum(a!='H' for a in atoms),transition_type=kind,special=special,reference_basis='aug-cc-pVTZ')
  label=dict(tbe_ev=r['TBE/AVTZ'],method=r.get('Method'),safe_flag=r.get('Safe ? (~50 meV)'))
  candidates[family].append((public,label))
rng=random.Random(20261030);selected=[];labels={};counts={}
for family,rows in candidates.items():
 # Round-robin across molecule names avoids a single molecule filling a family.
 groups={}
 for public,label in rows:groups.setdefault(public['molecule'],[]).append((public,label))
 names=sorted(groups);rng.shuffle(names)
 for name in names:rng.shuffle(groups[name])
 chosen=[]
 while len(chosen)<12 and any(groups.values()):
  for name in names:
   if groups[name] and len(chosen)<12:chosen.append(groups[name].pop())
 counts[family]=dict(eligible_states=len(rows),selected=len(chosen),shortfall=max(0,12-len(chosen)))
 for public,label in chosen:
  sid='B%03d'%(len(selected)+1);public['id']=sid;selected.append(public);labels[sid]=label
  target=OUT/'geometries'/f'{sid}.xyz';target.parent.mkdir(exist_ok=True);target.write_bytes((Q/public['geometry_source']).read_bytes())
sealed=OUT/'sealed_labels.json';write(sealed,labels);sealed.chmod(0o400)
manifest=dict(schema='mouse.blind_validation.manifest.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.getfqdn(),rules_sha256=sha(P/'protocol_v1.json'),candidate_source_sha256=sha(P/'inputs/candidate_source_protocol.json'),reference_commit=subprocess.check_output(['git','-C',str(Q),'rev-parse','HEAD'],text=True).strip(),exposure_audit_sha256=sha(OUT/'exposure_audit.json'),labels_sha256=sha(sealed),reference_labels_segregated=True,energy_window_ev=20.0,selection_seed=20261030,counts=counts,states=selected,limitation='Blinded relative to numerical labels in the prediction workflow and audited MOUSE exposure; not an external human-held prospective trial')
write(OUT/'manifest.json',manifest);(OUT/'manifest.sha256').write_text(sha(OUT/'manifest.json')+'\n')
print(json.dumps(dict(selected=len(selected),counts=counts,manifest_sha256=sha(OUT/'manifest.json'),selected_molecules=sorted(set(r['molecule'] for r in selected))),indent=2))
