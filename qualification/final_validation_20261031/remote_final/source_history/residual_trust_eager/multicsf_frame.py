"""Regular fixed-relative-active-frame manifold for two multi-CSF parents.

Preserves the FULL active cross-overlap matrix, a stricter and different PES
than individual cosine constraints. A common metric-orthogonal rotation moves
both frames, while independent spectator rotations preserve active columns.
An SVD removes energy-gauge redundancy from this explicit parameterization;
no singular cosine constraint or second-order condition is dropped.
"""
import numpy as np
from scipy.linalg import block_diag,expm

class RelativeActiveFrame:
    def __init__(self,objects):
        self.objects=objects;self.n=len(objects[0].context.state.overlap);self.s=objects[0].context.state.overlap
        self.active=np.r_[np.arange(objects[0].rank),objects[0].no+np.arange(objects[0].rank)]
        self.left,self.right=np.triu_indices(self.n,1);self.ng=len(self.left)
        self.generators=np.zeros((self.ng,self.n,self.n));self.generators[np.arange(self.ng),self.left,self.right]=1;self.generators[np.arange(self.ng),self.right,self.left]=-1
        self.private=[np.flatnonzero(~np.isin(o.left,self.active)&~np.isin(o.right,self.active)) for o in objects]
        self.offset=np.cumsum([self.ng]+[len(p) for p in self.private])
    def maps(self,point):
        v=point[0].T@self.s@point[1];gs=[self.generators,v.T@self.generators@v]
        raw=[]
        for side,o in enumerate(self.objects):
            m=np.zeros((len(o.left),self.offset[-1]));m[:,:self.ng]=gs[side][:,o.left,o.right].T
            # offset[0]=ng; offset[1]=ng+np0, offset[2]=total.
            start=self.ng if side==0 else self.offset[1]
            m[self.private[side],start+np.arange(len(self.private[side]))]=1;raw.append(m)
        joined=np.concatenate(raw,axis=0);u,sv,vt=np.linalg.svd(joined,full_matrices=False);keep=sv>1e-9
        return u[:,keep],vt[keep].T/sv[keep],v
    def retract(self,point,step,rawmap,v):
        raw=rawmap@step;common=expm(np.einsum('a,aij->ij',raw[:self.ng],self.generators));result=[]
        for side,(o,c,p) in enumerate(zip(self.objects,point,self.private)):
            start=self.ng if side==0 else self.offset[1];k=np.zeros((self.n,self.n));k[o.left[p],o.right[p]]=raw[start:start+len(p)];k[o.right[p],o.left[p]]=-k[o.left[p],o.right[p]]
            result.append(point[0]@common@(np.eye(self.n) if side==0 else v)@expm(k))
        return result
    def evaluate(self,point):
        rows=[o.evaluate(c,True) for o,c in zip(self.objects,point)];t,rawmap,v=self.maps(point)
        g=np.concatenate([.5*r[1] for r in rows]);h=block_diag(*[.5*r[2] for r in rows])
        # Exact BCH mixed term of exp(common) exp(private). It vanishes
        # at independently stationary anchors but is essential off anchor.
        extra=np.zeros((self.offset[-1],self.offset[-1]))
        for side,(o,row,private) in enumerate(zip(self.objects,rows,self.private)):
            gg=np.zeros((self.n,self.n));gg[o.left,o.right]=row[1];gg[o.right,o.left]=-row[1]
            hs=self.generators if side==0 else v.T@self.generators@v
            comm=gg@hs-hs@gg
            correction=.25*comm[:,o.left[private],o.right[private]]
            # The common rotation also contains spectator gauge generators.
            # Their energy gradient vanishes, but their mixed exponential
            # curvature with active rotations does not vanish off stationarity.
            local=np.zeros((self.offset[-1],self.n,self.n));local[:self.ng]=hs
            private_start=self.ng if side==0 else self.offset[1]
            for k,pair in enumerate(private):
                i,j=o.left[pair],o.right[pair];local[private_start+k,i,j]=1;local[private_start+k,j,i]=-1
            active=np.zeros_like(local);active[:,o.left,o.right]=local[:,o.left,o.right];active[:,o.right,o.left]=-active[:,o.left,o.right]
            gauge=local-active
            bracket=.5*np.einsum('aij,bij->ab',gg@active-active@gg,gauge,optimize=True)
            extra-=.25*(bracket+bracket.T)
            columns=(self.ng if side==0 else self.offset[1])+np.arange(len(private))
            extra[np.ix_(np.arange(self.ng),columns)]+=correction
            extra[np.ix_(columns,np.arange(self.ng))]+=correction.T
        hh=t.T@h@t+rawmap.T@extra@rawmap
        return sum(.5*r[0] for r in rows),t.T@g,.5*(hh+hh.T),t,rawmap,v
    def polish(self,point):
        """Solve frame stationarity with a safeguarded residual trust step.

        Uniformly clipping a soft-mode Newton step also suppresses stiff-mode
        corrections. The spectral trust solve minimizes the predicted residual
        in a rotation ball instead. State and full-frame certificates are kept.
        """
        target=point[0][:,self.active].T@self.s@point[1][:,self.active]
        for iteration in range(64):
            e,g,h,t,rawmap,v=self.evaluate(point)
            if np.max(abs(g))<1e-10:return point
            old_ci=[o.cache['ci'].copy() for o in self.objects]
            values,vectors=np.linalg.eigh(h);projected=vectors.T@g
            step=-vectors@np.divide(projected,values,out=np.zeros_like(projected),where=abs(values)>1e-14)
            def residual_step(radius):
                square=values**2
                def trial(mu):return -vectors@(values*projected/(square+mu))
                low=0.;high=max(1.,float(np.max(square)))
                while np.linalg.norm(trial(high))>radius:high*=2
                for unused in range(60):
                    middle=.5*(low+high)
                    if np.linalg.norm(trial(middle))>radius:low=middle
                    else:high=middle
                return trial(high)
            if np.linalg.norm(step)>.3:step=residual_step(.3)
            accepted=False
            for direction in [step,residual_step(.03)]:
                for fraction in [1.,.5,.25,.125,.0625,.03125]:
                    trial=self.retract(point,fraction*direction,rawmap,v);_,gg,_,_,_,_=self.evaluate(trial)
                    guard=all(o.fidelity(c)[0]**2>.5 and o.cache['coefficient_margin']>=.05 and abs(o.signed_overlap(old,ci,c,o.cache['ci']))>=.9 for o,old,ci,c in zip(self.objects,point,old_ci,trial))
                    if guard and np.linalg.norm(gg)<np.linalg.norm(g):point=trial;accepted=True;break
                if accepted:break
            if not accepted:raise RuntimeError('frame continuation blocked; checkpoint not accepted')
            assert np.max(abs(point[0][:,self.active].T@self.s@point[1][:,self.active]-target))<1e-10
        raise RuntimeError('frame continuation budget exceeded')
    def anchor_response(self,point,responses):
        if max(np.max(abs(o.evaluate(c,True)[1])) for o,c in zip(self.objects,point))>1e-9:raise ValueError('independently stationary anchor required')
        return self.nuclear_response(point,responses)
    def nuclear_response(self,point,responses):
        e,g,h,t,rawmap,v=self.evaluate(point)
        if np.max(abs(g))>1e-9:raise ValueError('joint frame stationarity required')
        kr=t.T@np.concatenate([.5*r['orbital_nuclear_coupling'] for r in responses],axis=0);rr=sum(.5*r['frozen_hessian'] for r in responses)
        curvature=np.linalg.eigvalsh(h)
        if np.min(abs(curvature))<1e-7:raise ValueError('unresolved frame response curvature')
        response=np.linalg.solve(h,-kr);hh=rr+kr.T@response
        return dict(energy=e,gradient=sum(.5*r['gradient'] for r in responses),hessian=.5*(hh+hh.T),orbital_hessian=h,orbital_response=response,maximum_response_residual=float(np.max(abs(h@response+kr))),minimum_absolute_curvature=float(np.min(abs(curvature))),dimension=len(g),active_cross_overlap=point[0][:,self.active].T@self.s@point[1][:,self.active])
