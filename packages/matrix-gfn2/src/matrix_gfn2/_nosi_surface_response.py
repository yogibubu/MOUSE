"""Complete prepared spin-pure OO-CID matrix response, NumPy/SciPy only."""
import ast
from types import SimpleNamespace
import numpy as np
from ._mixed_jet import MixedJet as Jet,contract,stack
from ._nosi_mixed import inverse
from ._nosi_nuclear import closed,opened,csf_element,square_root_jet,DensityEnergy
from ._nosi_orbital_response import GroundResponse,ROKSResponse
from ._nosi_slater import Hamiltonian
from ._nosi_spin import spin_vectors,ASSIGNMENTS,matrix as spin_matrix
from ._nosi_configurations import basis as configuration_basis,common_core_and_active
from ._nosi_canonical import block as canonical_block

class MatrixResponse:
    def __init__(self,context,reference,ground_coefficients,state_coefficients,multipliers,canonical_frame,labels,h,eri,*,derivatives=True,auxiliary_records=None,scalar_elements=None,scalar_checkpoint=None,fixed_bra=False,coulomb_factors=None):
        if derivatives and getattr(context,'vertical_only',False):raise NotImplementedError('this quadrature adapter supports vertical energies only')
        self.context=context;self.s=context.state;self.reference=reference;self.no=context.state.electron_count//2
        self.nr=context.state.coordinates_bohr.size if derivatives else 0;self.labels=list(labels)
        self.ycanonical=np.asarray(canonical_frame);self.h=np.asarray(h);self.w=np.asarray(eri);self.nbase=1+2*len(state_coefficients)
        self.auxiliary_records=auxiliary_records
        self.scalar_elements={} if scalar_elements is None else scalar_elements
        self.scalar_checkpoint=scalar_checkpoint
        self.scalar=Hamiltonian(self.h,self.w);self.scalar.coulomb_factors=coulomb_factors;self.constant_elements={};self.spin_map=np.zeros((4,4,2,2))
        self.fixed_bra=None;self.canonical_block=None
        for i in range(4):
            for j in range(4):
                m=np.zeros((4,4));m[i,j]=1.;self.spin_map[i,j]=spin_matrix(0.,m)
        if auxiliary_records is not None:
            core,active=common_core_and_active(auxiliary_records)
            if fixed_bra and not derivatives:
                from ._nosi_fixed_bra import FixedBraCSF
                self.fixed_bra=FixedBraCSF(self.scalar,self.ycanonical[:,core],self.ycanonical[:,active])
            metric,matrix,_=canonical_block(None,self.ycanonical,self.h,self.w,core,active,records=auxiliary_records)
            self.canonical_block=(metric,matrix)
        if derivatives:
            self.ground=GroundResponse(context,ground_coefficients);self.states=[]
            for c,mu in zip(state_coefficients,multipliers):
                q=np.outer(ground_coefficients[:,:self.no].T@context.state.overlap@c[:,0],ground_coefficients[:,self.no:].T@context.state.overlap@c[:,self.no]);q/=np.linalg.norm(q)
                self.states.append(ROKSResponse(context,self.w,c,self.ground,mu,q))
            self.energy=DensityEnergy(context)
        else:
            occ=np.zeros((2,len(ground_coefficients)));occ[:,:self.no]=1
            geometry=(None,Jet.constant(context.state.overlap,0),None,None,Jet.constant(context.charge_correction.transport.u,0))
            self.ground=SimpleNamespace(c=ground_coefficients,occupations=occ,coefficient=lambda direction:Jet.constant(ground_coefficients,0),engine=SimpleNamespace(geometry=lambda direction:geometry))
            mixed=occ.copy();mixed[0,0]=0;mixed[0,self.no]=1;triplet=np.array([np.maximum(*mixed),np.minimum(*mixed)])
            self.states=[SimpleNamespace(occupations=mixed,triplet=triplet,coefficient=lambda direction,c=c:Jet.constant(c,0)) for c in state_coefficients]
            class ValueEnergy:
                def evaluate(self,c,occupations,direction):
                    p=np.array([(c.v*x)@c.v.T for x in occupations]);return Jet.constant(context.energy(p) if hasattr(context,'energy') else context.fock_and_energy(p)[1],0)
            self.energy=ValueEnergy()
    def _spin_pair(self,left,right,direction):
        if hasattr(self.context,'spin_pair_bilinear'):return self.context.spin_pair_bilinear(left,right,direction)
        return contract('i,ij,j->',left,self.reference.pair_spin_hartree,right)
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
        four_overrides=[]
        if self.auxiliary_records is not None:
            basis.extend(configuration_basis(y,self.auxiliary_records))
            pending=None;q_cache={};pair_cache={}
            for i,record in enumerate(self.auxiliary_records):
                occupation=np.zeros_like(self.ground.occupations);core=record['closed'];sites=record.get('open',[]);occupation[:,core]=1
                if record['kind']=='closed':diagonal.append(self.energy.evaluate(c,occupation,direction))
                elif record['kind']=='open':
                    occupation[0,sites[0]]=1;occupation[1,sites[1]]=1
                    triplet=np.zeros_like(occupation);triplet[:,core]=1;triplet[0,sites]=1
                    diagonal.append(2*self.energy.evaluate(c,occupation,direction)-self.energy.evaluate(c,triplet,direction))
                else:
                    if record['component']==0:
                        occupation[:,sites]=.5;charge=self.energy.evaluate(c,occupation,direction);q=[]
                        for site in sites:
                            if site not in q_cache:
                                density=contract('i,j->ij',c[:,site],c[:,site]);pulled=u.T@(.5*(s@density+density@s))@u
                                q_cache[site]=contract('aij,ij->a',self.reference.operators,pulled)
                            q.append(q_cache[site])
                        pending=Jet.constant(np.zeros((2,2)),self.nr)
                        for j in range(4):
                            for k in range(4):
                                key=tuple(sorted((sites[j],sites[k])))
                                if key not in pair_cache:pair_cache[key]=self._spin_pair(q[j],q[k],direction)
                                pending=pending+pair_cache[key][...,None,None]*self.spin_map[j,k]
                        pending=pending+charge[...,None,None]*np.eye(2);four_overrides.append((self.nbase+i,pending))
                    diagonal.append(pending[record['component'],record['component']])
        else:
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
                    value=self._spin_pair(q[i],q[j],direction);four=four+value[...,None,None]*self.spin_map[i,j]
            four=four+charge[...,None,None]*np.eye(2);diagonal.extend([four[0,0],four[1,1]])
            four_overrides.append((len(basis)-2,four))
        # Retained nonorthogonal CSFs repeatedly use the same transition
        # densities. Batch their exact tensor actions before scalar assembly.
        if self.nr==0 and self.fixed_bra is not None:
            scalar_basis=[[(weight,(a.v,b.v)) for weight,(a,b) in csf] for csf in basis[:self.nbase]]
            self.scalar.prefetch_csf_pairs((scalar_basis[i],scalar_basis[j])
                for i in range(self.nbase) for j in range(i+1) if (i,j) not in self.scalar_elements)
        size=len(basis);metric=Jet.constant(np.zeros((size,size)),self.nr);hf=Jet.constant(np.zeros((size,size)),self.nr)
        if self.canonical_block is not None:
            metric.v[self.nbase:,self.nbase:]=self.canonical_block[0]
            hf.v[self.nbase:,self.nbase:]=self.canonical_block[1]
        # Frozen auxiliary values have identically zero nuclear jets. Fill
        # their dense block directly, without allocating scalar jets per pair.
        for i in range(size):
            stop=min(i+1,self.nbase) if self.canonical_block is not None else i+1
            for j in range(stop):
                if i>=self.nbase and j>=self.nbase:
                    key=(i,j)
                    if key not in self.constant_elements:
                        scalar=lambda b:[(weight,(a.v,bb.v)) for weight,(a,bb) in b]
                        self.constant_elements[key]=self.scalar.csf(scalar(basis[i]),scalar(basis[j]))
                    ov,hval=self.constant_elements[key]
                    metric.v[i,j]=metric.v[j,i]=ov;hf.v[i,j]=hf.v[j,i]=hval
                    continue
                elif self.nr==0:
                    key=(i,j)
                    if key not in self.scalar_elements:
                        scalar=lambda b:[(weight,(a.v,bb.v)) for weight,(a,bb) in b]
                        if self.fixed_bra is not None and i>=self.nbase and j<self.nbase:
                            self.scalar_elements[key]=self.fixed_bra.element(scalar(basis[j]),scalar(basis[i]))
                        else:self.scalar_elements[key]=self.scalar.csf(scalar(basis[i]),scalar(basis[j]))
                        if self.scalar_checkpoint is not None:self.scalar_checkpoint(self.scalar_elements)
                    ov,hval=self.scalar_elements[key];overlap=Jet.constant(ov,0);value=Jet.constant(hval,0)
                else:overlap,value=csf_element(basis[i],basis[j],self.h,self.w)
                metric[i,j]=metric[j,i]=overlap;hf[i,j]=hf[j,i]=value
        indices=np.arange(size)
        residual=stack(diagonal)-hf[indices,indices]/metric[indices,indices]
        h=hf+.5*metric*(residual[:,None]+residual[None,:])
        for index,four in four_overrides:h[index:index+2,index:index+2]=four
        self.current_basis=basis
        return h,metric,diagonal[0]
