"""Analytic response of anchored auxiliary CISD configurations.

Auxiliary canonical orbitals are frozen in the co-rotating orthonormal AO
frame at model preparation. They are not re-canonicalized at displacements.
The optimized target ROKS orbitals retain full implicit second response.
"""
from pathlib import Path
import sys,json,ast
ROOT=Path('/home/enzo/MOUSE-frontier-cisd-20261101');NATIVE=Path('/home/enzo/MOUSE-native-reference-20261101')
sys.path.insert(0,'/home/enzo/MOUSE-cam-response-20261101')
sys.path.insert(0,'/home/enzo/MOUSE-nosi-20261101')
import predict
# Ensure nuclear_assembly obtains CAM helpers rather than M06 imports.
import nuclear_assembly as na
na.DEST=NATIVE
from nuclear_assembly import Jet,closed,opened,csf_element,square_root_jet
from elements_mixed import inverse,contract
from spin_four import csfs as scalar_four,spin_vectors,ASSIGNMENTS,matrix as spin_matrix
from fast_slater import Hamiltonian
import numpy as np
from scipy.linalg import eigh


def eigen_jet(h,s,root):
    e,c=eigh(h.v,s.v);energy=e[root];v=c[:,root];a=h.a-energy*s.a;b=h.b-energy*s.b
    g=np.einsum('i,aij,j->a',v,a,v);t=float(v@b@v)
    mixed=np.einsum('i,aij,j->a',v,h.ab-energy*s.ab,v)-t*np.einsum('i,aij,j->a',v,s.a,v)-g*float(v@s.b@v)
    for j in range(len(e)):
        if j==root:continue
        gap=energy-e[j]
        if abs(gap)<1e-8:raise ValueError('unresolved state degeneracy')
        mixed+=2*np.einsum('i,aij,j->a',c[:,j],a,v)*float(c[:,j]@b@v)/gap
    return Jet(np.array(energy),g,np.array(t),mixed)

class Candidate(na.NuclearCandidate):
    def __init__(self,cohort,name):
        super().__init__(cohort,name)
        self.destination=ROOT/'nosi_analytic'/cohort/name;self.destination.mkdir(parents=True,exist_ok=True)
        report=json.loads((ROOT/cohort/name/'result.json').read_text());self.labels=report['labels'];self.nbase=1+2*len(self.states)
        self.roots={r['id']:r['root'] for r in report['predictions']};self.vertical={r['id']:r['gap_ev'] for r in report['predictions']}
        f,_=self.context.fock_and_energy(np.array([self.ground.c[:,:self.no]@self.ground.c[:,:self.no].T]*2))
        _,self.canonical=eigh(f[0],self.s.overlap);w,u=eigh(self.s.overlap);self.ycanonical=(u*np.sqrt(w))@u.T@self.canonical
        self.scalar=Hamiltonian(self.h,self.w);self.constant_elements={}
        self.spin_map=np.zeros((4,4,2,2))
        for i in range(4):
            for j in range(4):
                m=np.zeros((4,4));m[i,j]=1;self.spin_map[i,j]=spin_matrix(0.,m)
    def matrices(self,direction):
        direction=np.asarray(direction).ravel();_,s,_,_,u=self.ground.engine.geometry(direction);root=square_root_jet(s)
        cg=self.ground.coefficient(direction);cs=[state.coefficient(direction) for state in self.states]
        yg=u.T@root@cg;ys=[u.T@root@c for c in cs]
        basis=[closed(yg,list(range(self.no)))];diagonal=[self.energy.evaluate(cg,self.ground.occupations,direction)]
        for c,y,state in zip(cs,ys,self.states):
            basis.append(opened(y,self.no));diagonal.append(2*self.energy.evaluate(c,state.occupations,direction)-self.energy.evaluate(c,state.triplet,direction))
        occ=self.ground.occupations.copy();occ[:,0]=0;occ[:,self.no]=1
        for c,y in zip(cs,ys):
            basis.append(closed(y,list(range(1,self.no))+[self.no]));diagonal.append(self.energy.evaluate(c,occ,direction))
        y=Jet.constant(self.ycanonical,self.nr);c=inverse(root)@u@y;core=list(range(self.no-2))
        for label in self.labels[self.nbase:-2]:
            occupation=np.zeros((2,len(self.s.overlap)))
            if label.startswith('pair_'):
                pair=ast.literal_eval(label[5:]);indices=core+list(pair);occupation[:,indices]=1
                basis.append(closed(y,indices));diagonal.append(self.energy.evaluate(c,occupation,direction))
            else:
                _,paired,opens=label.split('_',2);paired=int(paired);opens=ast.literal_eval(opens);indices=core+[paired]
                first=y[:,indices+[opens[0]]];second=y[:,indices+[opens[1]]]
                basis.append([(2**-.5,(first,second)),(2**-.5,(second,first))]);occupation[:,indices]=1;occupation[0,opens[0]]=1;occupation[1,opens[1]]=1
                triplet=np.zeros_like(occupation);triplet[:,indices]=1;triplet[0,list(opens)]=1
                diagonal.append(2*self.energy.evaluate(c,occupation,direction)-self.energy.evaluate(c,triplet,direction))
        vectors,phases=spin_vectors();sites=list(range(self.no-2,self.no+2))
        for column in range(2):
            csf=[]
            for row,alpha in enumerate(ASSIGNMENTS):
                weight=vectors[row,column]*phases[row]
                if weight:csf.append((float(weight),(y[:,core+[sites[i] for i in alpha]],y[:,core+[sites[i] for i in range(4) if i not in alpha]])))
            basis.append(csf)
        balanced=self.ground.occupations.copy();balanced[:,self.no-2:self.no+2]=.5;charge=self.energy.evaluate(c,balanced,direction)
        original=self.reference.operators
        q=[]
        for i in sites:
            density=contract('i,j->ij',c[:,i],c[:,i]);pulled=u.T@(.5*(s@density+density@s))@u
            q.append(contract('aij,ij->a',original,pulled))
        # Four-spin SU(2) completion of the identical quadratic spin functional.
        four=Jet.constant(np.zeros((2,2)),self.nr)
        for i in range(4):
            for j in range(4):
                value=contract('i,ij,j->',q[i],self.reference.pair_spin_hartree,q[j]);four=four+value[...,None,None]*self.spin_map[i,j]
        four=four+charge[...,None,None]*np.eye(2);diagonal.extend([four[0,0],four[1,1]])
        size=len(basis);metric=Jet.constant(np.zeros((size,size)),self.nr);hf=Jet.constant(np.zeros((size,size)),self.nr)
        for i in range(size):
            for j in range(i+1):
                if i>=self.nbase and j>=self.nbase:
                    key=(i,j)
                    if key not in self.constant_elements:
                        scalar=lambda b:[(weight,(a.v,bb.v)) for weight,(a,bb) in b]
                        self.constant_elements[key]=self.scalar.csf(scalar(basis[i]),scalar(basis[j]))
                    ov,hval=self.constant_elements[key];overlap=Jet.constant(ov,self.nr);value=Jet.constant(hval,self.nr)
                else:overlap,value=csf_element(basis[i],basis[j],self.h,self.w)
                metric[i,j]=metric[j,i]=overlap;hf[i,j]=hf[j,i]=value
        residual=[energy-hf[i,i]/metric[i,i] for i,energy in enumerate(diagonal)]
        h=Jet.constant(np.zeros((size,size)),self.nr)
        for i in range(size):
            for j in range(size):h[i,j]=hf[i,j]+.5*metric[i,j]*(residual[i]+residual[j])
        h[-2:,-2:]=four
        return h,metric,diagonal[0]

if __name__=='__main__':
    import argparse,time,resource
    parser=argparse.ArgumentParser();parser.add_argument('cohort');parser.add_argument('name');parser.add_argument('--column',type=int);args=parser.parse_args()
    with na.threadpool_limits():
        start=time.monotonic();candidate=Candidate(args.cohort,args.name)
        if args.column is not None:
            direction=np.eye(candidate.nr)[args.column];h,s,eg=candidate.matrices(direction);result={}
            for identity,root in candidate.roots.items():
                jet=eigen_jet(h,s,root);gap=float(jet.v-eg.v)*na.EV;assert abs(gap-candidate.vertical[identity])<1e-5,(identity,gap,candidate.vertical[identity]);result[identity]=dict(gap_ev=gap,energy_hartree=float(jet.v),gradient=jet.a,hessian_column=jet.ab)
            np.savez_compressed(candidate.destination/f'column_{args.column:03d}.npz',ground_gradient=eg.a,ground_hessian_column=eg.ab,**{i+'_gradient':x['gradient'] for i,x in result.items()},**{i+'_hessian_column':x['hessian_column'] for i,x in result.items()})
            na.write(candidate.destination/f'column_{args.column:03d}.json',dict(states=result,wall_seconds=time.monotonic()-start,source_sha256=na.sha(__file__),method='complete analytic anchored-auxiliary CISD response'))
        else:
            # Reuse the tested resume/checkpoint and invariance assembly loop.
            na.projected_eigen_jet=eigen_jet;result=candidate.hessian()
            na.write(candidate.destination/'hessian_certificate.json',dict(host=na.HOST,states={i:{k:v for k,v in d.items() if k not in ['gradient','hessian']} for i,d in result.items()},wall_seconds=time.monotonic()-start,peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=na.sha(__file__),protocol_sha256=na.sha(ROOT/'protocol.json')))
