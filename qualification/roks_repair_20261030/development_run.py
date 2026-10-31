"""Complete the fixed seven-molecule development panel without refitting."""
from remote_bootstrap import HOST,BASE
import json,traceback
import spin_kernel_probe as spin
import response_kernel_probe as full

for case in spin.plan():
 for job in (spin,full):
  path=job.ROOT/case['name']/'result.json'
  if path.exists() and len(json.loads(path.read_text())['records'])==len(case['targets']):
   print('REUSE',job.ROOT,case['name'],flush=True);continue
  try:
   job.write(job.ROOT/'protocol.json',job.PROTOCOL);job.run(case)
  except Exception as exc:
   job.write(job.ROOT/case['name']/'failure.json',dict(host=HOST,error=str(exc),traceback=traceback.format_exc()))
   print(traceback.format_exc(),flush=True)
