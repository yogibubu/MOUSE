"""Snapshot completed evidence and provenance, excluding sealed numerical labels."""
from pathlib import Path
import socket,hashlib,json,tarfile,io,datetime
assert socket.getfqdn()=='oracle.sns.it'
R=Path(__file__).resolve().parent;entries=[];omitted=[]
now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=R/f'evidence_{now}.tar.gz'
with tarfile.open(out,'w:gz') as tar:
 for p in sorted(R.rglob('*')):
  if not p.is_file() or p.is_symlink() or p.suffix in ['.gz','.pyc'] or p.name in ['sealed_labels.json','controller.lock','compute.lock'] or '__pycache__' in p.parts:continue
  size=p.stat().st_size
  if size>10*1024**2 or p.suffix=='.chk' or ('portable_model' in p.parts and p.name.endswith('.npz')) or ('anchor' in p.parts and p.name.endswith('.npz')):
   h=hashlib.sha256()
   with p.open('rb') as stream:
    for data in iter(lambda:stream.read(1024**2),b''):h.update(data)
   omitted.append(dict(path=str(p.relative_to(R)),sha256=h.hexdigest(),size=size,retained_remotely=True));continue
  if p.suffix not in ['.json','.py','.npz','.txt','.log','.resources','.xyz','.xml']:continue
  data=p.read_bytes();name=str(p.relative_to(R));info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data));entries.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest(),size=len(data)))
 manifest=dict(host=socket.getfqdn(),snapshot_utc=now,in_progress_snapshot=not all(json.loads((R/'final_results.json').read_text())['operational_closure'].values()),archived=entries,large_files_retained_remotely=omitted,segregated_label_files_included=False,published_score_reference_energies_included=True)
 data=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('TRANSFER.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
print(json.dumps(dict(archive=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),files=len(entries),remote_large_files=len(omitted))))
