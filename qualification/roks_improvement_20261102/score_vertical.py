"""Score only qualified predictions; never imported by the electronic driver."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
protocol=json.loads((root/'protocol.json').read_text())
# Historically exposed QUEST targets, same values as the committed ledger.
references={'B001':3.966,'B002':7.897,'B003':6.222,'V010':6.530,'N001':4.304,'V001':4.468}
rows=[];summary={}
for case in protocol['cases']:
 for functional in protocol['functionals']:
  path=root/'vertical'/case['id']/functional/'result.json'
  if not path.exists():continue
  data=json.loads(path.read_text())
  assert data['orbital_residual']<1e-8 and min(data['hole_overlap'],data['particle_overlap'])>=.7
  rows.append(dict(id=case['id'],name=case['name'],role=case['role'],functional=functional,
     reference_ev=references[case['id']],prediction_ev=data['excitation_ev'],error_ev=data['excitation_ev']-references[case['id']]))
for role in ['pilot','confirmation']:
 expected=sum(case['role']==role for case in protocol['cases']);summary[role]={}
 for functional in protocol['functionals']:
  errors=[row['error_ev'] for row in rows if row['role']==role and row['functional']==functional]
  summary[role][functional]=dict(completed=len(errors),expected=expected,
     MAE_ev=sum(abs(e) for e in errors)/expected if len(errors)==expected else None,
     maximum_error_ev=max(map(abs,errors)) if len(errors)==expected else None)
ranked=all(x['MAE_ev'] is not None for x in summary['pilot'].values())
selected=min(summary['pilot'],key=lambda f:summary['pilot'][f]['MAE_ev']) if ranked else None
report=dict(label_exposure='Historically exposed, not blind',basis_limitation='aug-cc-pVDZ calculations vs QUEST TBE/AVTZ targets',
     rows=rows,summary=summary,selected_global_functional=selected,offset_ev=0.,energy_based_root_selection=False)
(root/'functional_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
