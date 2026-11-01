"""Postprocess saved CI vectors against the frozen native-frontier targets."""
from run_pigment_cid import *

def assign(name):
 folder=HERE/'pigment_cid'/name;path=folder/'result.json';report=json.loads(path.read_text());roks=json.loads((HERE/'pigment_roks'/name/'result.json').read_text())
 with np.load(HERE/'pigment_models'/name/'state.npz') as z:s=z['overlap'];c=z['coefficients'];no=int(z['electron_count'])//2
 n=len(c);values,u=eigh(s);root=(u*values**.5)@u.T;ordinals=report.get('retained_ROKS_ordinals',[row['ordinal'] for row in roks['states']]);ys=[]
 for ordinal in ordinals:
  with np.load(HERE/'pigment_roks'/name/f'state_{ordinal}/checkpoint.npz') as z:ys.append(root@z['coefficients'])
 basis=scalar_basis(root@c,ys,report['auxiliary_records'],no);targets=[]
 for row in roks['states']:
  h,p=row['hole'],row['particle'];order=[h]+[j for j in range(no) if j!=h]+[p]+[j for j in range(no,n) if j!=p];targets.append(opened(root@c[:,order],no))
 cross=np.array([[overlap(a,b) for b in basis] for a in targets])
 with np.load(folder/'eigenstates.npz') as z:weights=np.sum((cross@z['ci'])**2,axis=0)
 selected=np.sort(np.argsort(weights)[-len(targets):]);assert weights[selected].min()>=.49,('Unresolved frontier assignment',weights)
 previous=folder/'initial_ROKS_subspace_assignment.json'
 if not previous.exists():write(previous,report)
 for i,row in enumerate(report['states']):row.update(native_frontier_subspace_weight=float(weights[i]),selected=bool(i in selected))
 report['selected_states']=[report['states'][i] for i in selected];report['retained_ROKS_ordinals']=ordinals;report['target_rule']='Largest collective overlap with the frozen native-frontier singlet subspace, minimum weight 0.49; ROKS weights reported separately';write(path,report);print(name,report['selected_states'],flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('name');args=parser.parse_args()
 with threadpool_limits(2):assign(args.name)
