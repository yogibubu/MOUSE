from remote_bootstrap import BASE
import json,numpy as np
from pathlib import Path
for c in json.loads((BASE/'protocol.json').read_text())['cases']:
 folder=Path(c['reference']);d=json.loads((folder/'result.json').read_text())
 print(c['name'],'result keys',list(d))
 if 'assignments' in d:print('assign',str(d['assignments'])[:850])
 for p in folder.glob('*.npz'):
  with np.load(p,allow_pickle=False) as f:print(p.name,{k:f[k].shape for k in f.files if 'amplit' in k or 'vector' in k or 'energ' in k})
