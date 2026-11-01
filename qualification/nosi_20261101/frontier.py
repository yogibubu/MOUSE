"""Recover prepared stationary states through the ground-program frontend."""
from pathlib import Path
import sys,argparse
parser=argparse.ArgumentParser();parser.add_argument('functional',choices=['m06','cam']);parser.add_argument('cohort');parser.add_argument('molecule');args=parser.parse_args()
ROOT=Path('/home/enzo/MOUSE-'+args.functional+'-response-20261101');sys.path.insert(0,str(ROOT))
from predict import *
from native_batch import install_native_batch

def singlet_overlap(c,d,s,no):
    ia=list(range(1,no))+[no];ib=list(range(no));indices=[(ia,ib),(ib,ia)];total=0.
    for aa,ab in indices:
        for ba,bb in indices:total+=.5*np.linalg.det(c[:,aa].T@s@d[:,ba])*np.linalg.det(c[:,ab].T@s@d[:,bb])
    return abs(float(total))

def run(cohort,name):
    out=ROOT/cohort/name;destination=out/'frontier_ground_program';destination.mkdir(exist_ok=True)
    case=prepare(cohort,name);s=case['s'];no=case['no'];n=len(s.overlap);install_native_batch(case,out)
    rows=[row for row in json.loads((OUT/cohort/'manifest.json').read_text())['states'] if row['molecule']==name]
    forbidden={str((case['ref']/(row['id']+'.npz')).resolve()) for row in rows};original_load=np.load;reads=[]
    def safe_load(path,*a,**kw):
        if isinstance(path,(str,Path)):
            resolved=str(Path(path).resolve());assert resolved not in forbidden and 'backend_tda' not in resolved
            reads.append(resolved)
        return original_load(path,*a,**kw)
    np.load=safe_load
    scf.hf.kernel=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('DFT SCF forbidden in frontier recovery'))
    case['calc'].evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC forbidden in frontier recovery'))
    model=out/'coherent_common/model.npz'
    with np.load(model) as z:data={k:z[k].copy() for k in ['operators','charge','spin','q0','linear']}
    reference=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,data,sha(model))
    context=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=reference)
    ground_path=out/'coherent_common/ground/stationary.npz'
    if not ground_path.exists():ground_path=out/'coherent_continuation_models/t_1.00000000/ground/stationary.npz'
    with np.load(ground_path) as z:fg=z['result_fock'][0].copy();eg=float(z['result_energy_hartree'])
    levels,cg=eigh(fg,s.overlap);candidates=[]
    for hole in [no-1,no-2]:
        for particle in [no,no+1]:
            if hole<0 or particle>=n:continue
            folder=destination/f'h_{hole}_p_{particle}'
            if (folder/'result.json').exists():result=json.loads((folder/'result.json').read_text())
            else:
                order=[hole]+[i for i in range(no) if i!=hole]+[particle]+[i for i in range(no,n) if i!=particle]
                q=np.zeros((no,n-no));q[hole,particle-no]=1
                result=solve_stationary(context,case['eri'],no,cg[:,order],cg,q,folder)
            if result['qualified']:
                with np.load(folder/'final.npz') as z:c=z['coefficients'].copy()
                candidates.append((folder,c,result))
    matching=np.zeros((len(rows),len(candidates)));retained=[]
    for i,row in enumerate(rows):
        with np.load(out/'coherent_states'/row['id']/'final/final.npz') as z:c=z['coefficients'].copy()
        report=json.loads((out/'coherent_states'/row['id']/'result.json').read_text());retained.append(report)
        for j,(_,d,_) in enumerate(candidates):matching[i,j]=singlet_overlap(c,d,s.overlap,no)
    ii,jj=linear_sum_assignment(-matching);results=[]
    for i,j in zip(ii,jj):
        folder,c,result=candidates[j];difference=abs(result['energy_hartree']-retained[i]['energy_hartree'])*EV;overlap=float(matching[i,j])
        results.append(dict(id=rows[i]['id'],promotion=folder.name,CSF_overlap=overlap,energy_difference_ev=difference,qualified=bool(overlap>=.95 and difference<1e-5)))
    write(destination/'result.json',dict(states=results,all_recovered=len(results)==len(rows) and all(r['qualified'] for r in results),
        qualified_promotions=len(candidates),attempted_promotions=4,TDA_input_reads=0,new_SCF_calls=0,energy_labels_used=False,
        read_paths=sorted(set(reads)),source_sha256=sha(__file__),protocol_sha256=sha(Path(__file__).parent/'frontier_protocol.json')))
    print(name,results,flush=True)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():run(args.cohort,args.molecule)
