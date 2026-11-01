"""Relative-active-frame stationarity with two exact ground-overlap constraints."""
from anchor import *
sys.path.insert(0,'/home/enzo/MOUSE-transferability-20261101')
from orthogonal_singlet import ground_overlap_jet,ground_overlap_hessian
from matrix_gfn2.multicsf_frame import RelativeActiveFrame

class OverlapObjective:
    def __init__(self,base,ground,enabled):
        self.__dict__.update(vars(base));self.base=base;self.ground=ground;self.enabled=enabled
    def evaluate(self,c,curvature=True):
        if not self.enabled:return 0.,np.zeros(len(self.left)),np.zeros((len(self.left),len(self.left)))
        occ=self.base.occupations[0]
        value,g=ground_overlap_jet(self.ground,c,self.context.state.overlap,occ,self.left,self.right)
        h=ground_overlap_hessian(self.ground,c,self.context.state.overlap,occ,self.left,self.right)
        # RelativeActiveFrame averages its two constituents with weight one half.
        return 2*value,2*g,2*h

def overlap_frames(objects,grounds):
    return [RelativeActiveFrame([OverlapObjective(o,g.coefficients[0] if hasattr(g,'coefficients') else g,j==i) for j,(o,g) in enumerate(zip(objects,grounds))]) for i in range(2)]

def solve_joint(objects,grounds,point,out):
    out.mkdir(parents=True,exist_ok=True);frame=RelativeActiveFrame(objects);constraints=overlap_frames(objects,grounds)
    target=point[0][:,frame.active].T@frame.s@point[1][:,frame.active]
    history=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
    checkpoint=out/'checkpoint.npz';mu=np.zeros(2)
    if checkpoint.exists():
        with np.load(checkpoint) as a:point=[x.copy() for x in a['coefficients']];mu=a['multipliers'].copy()
    def evaluate(point,mu):
        e,g,h,t,rawmap,v=frame.evaluate(point);rows=[f.evaluate(point) for f in constraints]
        assert all(np.max(abs(r[3]-t))<1e-9 for r in rows)
        values=np.array([r[0] for r in rows]);jac=np.array([r[1] for r in rows]);ch=np.array([r[2] for r in rows])
        residual=np.r_[g-mu@jac,values]
        return e,g,h,values,jac,ch,residual,rawmap,v
    first=evaluate(point,mu);mu=np.linalg.lstsq(first[4].T,first[1],rcond=1e-10)[0]
    for iteration in range(100):
        e,g,h,values,jac,ch,res,rawmap,v=evaluate(point,mu)
        if np.max(abs(res))<1e-8:break
        system=np.block([[h-np.einsum('i,ijk->jk',mu,ch),-jac.T],[jac,np.zeros((2,2))]])
        step=np.linalg.lstsq(system,-res,rcond=1e-10)[0]
        length=np.linalg.norm(rawmap@step[:-2])
        if length>.25:step*=.25/length
        old_ci=[o.cache['ci'].copy() for o in objects];moved=False
        for fraction in [1.,.5,.25,.125,.0625,.03125,.015625]:
            trial=frame.retract(point,fraction*step[:-2],rawmap,v);mm=mu+fraction*step[-2:]
            *_,rr,_,_=evaluate(trial,mm)
            guard=all(o.fidelity(c)[1]>=.7 and abs(o.signed_overlap(old,ci,c,o.cache['ci']))>=.9 for o,old,ci,c in zip(objects,point,old_ci,trial))
            if guard and np.linalg.norm(rr)<np.linalg.norm(res):point=trial;mu=mm;moved=True;break
        history.append(dict(iteration=len(history),energy=e,kkt_max=float(np.max(abs(res))),ground_overlaps=values,accepted_step=moved))
        write(out/'history.json',history);np.savez_compressed(checkpoint,coefficients=np.array(point),multipliers=mu)
        if not moved:break
    e,g,h,values,jac,ch,res,rawmap,v=evaluate(point,mu)
    fidelities=[o.fidelity(c)[1] for o,c in zip(objects,point)]
    drift=float(np.max(abs(point[0][:,frame.active].T@frame.s@point[1][:,frame.active]-target)))
    result=dict(qualified=bool(np.max(abs(res))<1e-8 and min(fidelities)>=.7 and drift<1e-9),energy_hartree=e,
        parent_energies=[o.evaluate(c,False)[0] for o,c in zip(objects,point)],kkt_max=float(np.max(abs(res))),
        ground_overlaps=values,state_fidelities=fidelities,active_frame_drift=drift,iterations=len(history),multipliers=mu,
        source_sha256=sha(__file__),host=HOST,threads=2,nuclear_response_implemented=False)
    write(out/'result.json',result);np.savez_compressed(out/'final.npz',coefficients=np.array(point))
    return result,point
