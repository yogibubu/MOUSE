"""Transfer and verify the finite final Oracle evidence snapshot."""
from pathlib import Path
import json,subprocess,hashlib,tarfile
r=Path(__file__).resolve().parent
remote='/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration/final_validation'
meta=json.loads(subprocess.check_output(['ssh','oracle','cat',remote+'/final_archive.json'],text=True))
archive=r/Path(meta['archive']).name
subprocess.run(['scp','oracle:'+meta['archive'],str(archive)],check=True)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==meta['sha256']
out=r/'remote_final';out.mkdir(exist_ok=True)
with tarfile.open(archive) as tar:
 for member in tar.getmembers():
  assert member.isfile() and (out/member.name).resolve().is_relative_to(out.resolve())
 tar.extractall(out,filter='data')
manifest=json.loads((out/'TRANSFER.json').read_text());assert not manifest['in_progress_snapshot']
for row in manifest['archived']:
 p=out/row['path'];assert p.stat().st_size==row['size']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['path']
report=dict(status='pass',host=manifest['host'],archive=archive.name,archive_sha256=meta['sha256'],verified_files=len(manifest['archived']),large_files_retained_remotely=len(manifest['large_files_retained_remotely']),snapshot_utc=manifest['snapshot_utc'],in_progress_snapshot=False)
(r/'FINAL_TRANSFER_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
# Retain the verified extracted evidence and remote archive checksum; avoid a
# second binary copy in git. The remote tarball is retained for recovery.
archive.unlink()
print(json.dumps(report,indent=2))
