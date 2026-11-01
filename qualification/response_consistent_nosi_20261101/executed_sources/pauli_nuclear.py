"""Complete analytic anchored Pauli-NOSI nuclear response; no TDA derivative."""
from pathlib import Path
import sys,json,hashlib,time,resource,argparse
ROOT=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-response-consistent-nosi-20261101');CISD=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-frontier-cisd-20261101')
sys.path.insert(0,str(CISD))
import nuclear as prior
from nuclear import Candidate as CISDCandidate
import nuclear_assembly as na
from matrix_gfn2.state_interaction import Hamiltonian
import numpy as np

class Candidate(CISDCandidate):
    def __init__(self,cohort,name):
        super().__init__(cohort,name)
        self.destination=ROOT/'pauli_analytic'/cohort/name;self.destination.mkdir(parents=True,exist_ok=True)
        source=ROOT/'pauli_common'/cohort/name
        with np.load(source/'operators.npz') as z:self.h=z['h'].copy();self.w=z['eri'].copy()
        self.scalar=Hamiltonian(self.h,self.w);self.constant_elements={}
        result=json.loads((source/'result.json').read_text());self.roots={r['id']:r['root'] for r in result['primary']['predictions']};self.vertical={r['id']:r['gap_ev'] for r in result['primary']['predictions']}
        files=[Path(__file__),Path(prior.__file__),Path(na.__file__),source/'operators.npz',self.out/'coherent_common/model.npz',ROOT/'pauli_protocol.json']
        identity={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};path=self.destination/'response_identity.json'
        if path.exists():assert json.loads(path.read_text())==identity,'Resume identity changed; preserve the old checkpoint before a new model calculation'
        else:path.write_text(json.dumps(identity,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');p.add_argument('--column',type=int);a=p.parse_args()
    with na.threadpool_limits():
        start=time.monotonic();model=Candidate(a.cohort,a.name)
        if a.column is None:
            result=model.hessian()
            na.write(model.destination/'hessian_certificate.json',dict(host=na.HOST,states={i:{k:v for k,v in d.items() if k not in ['gradient','hessian']} for i,d in result.items()},wall_seconds=time.monotonic()-start,peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=na.sha(__file__),protocol_sha256=na.sha(ROOT/'pauli_protocol.json')))
        else:
            direction=np.eye(model.nr)[a.column];h,s,eg=model.matrices(direction);result={}
            for i,root in model.roots.items():
                jet=na.projected_eigen_jet(h,s,root);gap=float(jet.v-eg.v)*na.EV;assert abs(gap-model.vertical[i])<1e-5,(i,gap,model.vertical[i]);result[i]=dict(gap_ev=gap,energy_hartree=float(jet.v),gradient=jet.a,hessian_column=jet.ab)
            np.savez_compressed(model.destination/f'column_{a.column:03d}.npz',ground_gradient=eg.a,ground_hessian_column=eg.ab,**{i+'_gradient':x['gradient'] for i,x in result.items()},**{i+'_hessian_column':x['hessian_column'] for i,x in result.items()})
            na.write(model.destination/f'column_{a.column:03d}.json',dict(states=result,wall_seconds=time.monotonic()-start,source_sha256=na.sha(__file__)))
