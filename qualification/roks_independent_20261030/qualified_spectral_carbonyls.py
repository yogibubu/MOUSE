"""Already exposed B carbonyls: diagnostic transfer of the frozen spectral profile."""
from remote_bootstrap import HOST,BASE
import json
from threadpoolctl import threadpool_limits
import qualified_spectral_response as profile
import full_response as full
from spin_kernel_probe import write,digest
ROOT=BASE/'results/qualified_spectral_response'
protocol=dict(host=HOST,profile_sha256=digest(ROOT/'protocol.json'),profile_source_sha256=digest(BASE/'qualified_spectral_response.py'),
 cases=['acetaldehyde','acetone','formamide'],scope='previously exposed carbonyls; subsequent diagnostic, not a blind test',energy_offset=0.,lambda_exchange=.9,changes_to_profile=False)
p=ROOT/'carbonyl_protocol.json'
if p.exists():assert json.loads(p.read_text())==protocol
else:write(p,protocol)
rows=[]
with threadpool_limits(limits=2,user_api='blas'):
 full.ROOT=ROOT;full.calibrate=profile.calibrate
 for name in protocol['cases']:
  case=json.loads((BASE/'prepared_B'/name/'case.json').read_text());full.run(case)
  row=json.loads((ROOT/name/'n_pi/result.json').read_text());mapping=json.loads((ROOT/name/'mapping.json').read_text());t=case['targets'][0]
  row['target_orbital_mapping_qualified']=bool(mapping['qualified'][t['hole']] and mapping['qualified'][t['particle']]);row['qualified']=bool(row.get('qualified') and row['target_orbital_mapping_qualified']);rows.append(row)
write(ROOT/'carbonyl_summary.json',dict(host=HOST,protocol_sha256=digest(p),scope=protocol['scope'],rows=rows))
print('CARBONYLS COMPLETE',[(r['molecule'],r.get('qualified'),r.get('excitation_ev')) for r in rows],flush=True)
