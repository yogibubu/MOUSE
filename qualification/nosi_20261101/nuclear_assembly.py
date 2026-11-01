"""Complete analytic nuclear response of the fixed projected-NOSI candidate.

Frozen physical h/ERI couplings co-rotate in an orthonormal AO frame, just as
the calibrated one-body/response operators. Native diagonal energies remain
the full semiempirical functional. Moving-ground and ROKS second KKT response
is included; the generalized eigenvector and metric response is analytic.
"""
from pathlib import Path
import sys
sys.path.insert(0,'/home/enzo/MOUSE-m06-response-20261101')
from predict import *
from native_batch import install_native_batch
from orbital_mixed import GroundResponse,ROKSResponse,densities
from elements_mixed import element as determinant_element,Jet
from matrix_gfn2.deltascf_analytic_hessian import _fixed_density_geometry,_spin_geometry,_add_charge_response
from matrix_gfn2.nonlocal_response import TransportedOperators
from scipy.linalg import solve_sylvester
import refine

def square_root_jet(s):
    w,u=eigh(s.v);root=(u*np.sqrt(w))@u.T
    first=np.array([solve_sylvester(root,root,x) for x in s.a]);tangent=solve_sylvester(root,root,s.b)
    mixed=np.array([solve_sylvester(root,root,x-a@tangent-tangent@a) for x,a in zip(s.ab,first)])
    return Jet(root,first,tangent,mixed)

def closed(c,indices):return [(1.,(c[:,indices],c[:,indices]))]
def opened(c,no):
    a=c[:,list(range(1,no))+[no]];b=c[:,:no]
    return [(2**-.5,(a,b)),(2**-.5,(b,a))]

def csf_element(a,b,h,w):
    result=[Jet.constant(0.,a[0][1][0].n),Jet.constant(0.,a[0][1][0].n)]
    for wa,aa in a:
        for wb,bb in b:
            values=determinant_element(aa,bb,h,w)
            result=[x+wa*wb*y for x,y in zip(result,values)]
    return result

def projected_eigen_jet(h,s,root):
    p=Jet.constant(np.eye(len(s.v))[:,1:],s.n);p[0,:]=p[0,:]-s[0,1:]/s[0,0]
    hp=p.T@h@p;sp=p.T@s@p
    e,c=eigh(hp.v,sp.v);root=int(root);energy=e[root];v=c[:,root]
    a=hp.a-energy*sp.a;b=hp.b-energy*sp.b
    gradient=np.einsum('i,aij,j->a',v,a,v);tangent=float(v@b@v)
    mixed=np.einsum('i,aij,j->a',v,hp.ab-energy*sp.ab,v)
    mixed-=tangent*np.einsum('i,aij,j->a',v,sp.a,v)+gradient*float(v@sp.b@v)
    for other in range(len(e)):
        if other==root:continue
        gap=energy-e[other]
        if abs(gap)<1e-8:raise ValueError('unresolved projected state-interaction degeneracy')
        left=np.einsum('i,aij,j->a',c[:,other],a,v);right=float(c[:,other]@b@v)
        mixed+=2*left*right/gap
    return Jet(np.array(energy),gradient,np.array(tangent),mixed)

class DensityEnergy:
    def __init__(self,context):self.context=context;self.base={};self.spin={}
    def evaluate(self,c,occupations,direction):
        context=self.context;dm=densities(c,occupations);p=np.array([x.v for x in dm]);total=p.sum(0)
        key=(c.v.tobytes(),occupations.sum(0).tobytes())
        if key not in self.base:
            kernel,ds,df,d2s,rr=_fixed_density_geometry(context,p)
            if context.charge_correction is not None:
                charge=context.charge_correction;field,hessian=charge.geometry(total,ds,d2s);df=df+field;rr=rr+hessian;kernel=_add_charge_response(kernel,charge)
            self.base[key]=(kernel,ds,df,d2s,rr)
        kernel,ds,df,d2s,rr=self.base[key];spin_key=p.tobytes()
        if spin_key not in self.spin:
            sf,srr=_spin_geometry(context.polarization,p,ds,d2s)
            f,e=context.fock_and_energy(p);g=context.analytic_gradient(p,np.zeros_like(total)).ravel()
            self.spin[spin_key]=(df[:,None]+sf,rr+srr,f,e,g)
        explicit,curvature,f,e,g=self.spin[spin_key]
        first=np.array([x.a for x in dm]).transpose(1,0,2,3);tangent=np.array([x.b for x in dm]);mixed=np.array([x.ab for x in dm]).transpose(1,0,2,3)
        total_field=explicit.transpose(1,2,3,0)@direction
        total_field+=kernel(tangent.sum(0))[None]+context.polarization.response_batch(tangent[None])[0]
        derivative=g+np.einsum('sij,asij->a',f,first)
        second=curvature@direction+np.einsum('asij,sij->a',explicit,tangent)+np.einsum('sij,asij->a',total_field,first)+np.einsum('sij,asij->a',f,mixed)
        return Jet(np.array(e),derivative,np.array(direction@derivative),second)

class NuclearCandidate:
    def __init__(self,cohort,name):
        self.case=prepare(cohort,name);case=self.case;self.out=DEST/cohort/name;self.s=case['s'];self.no=case['no'];self.nr=self.s.coordinates_bohr.size
        self.destination=DEST/'nosi_analytic'/cohort/name;self.destination.mkdir(parents=True,exist_ok=True)
        install_native_batch(case,self.out)
        model=self.out/'coherent_common/model.npz'
        with np.load(model) as z:data={k:z[k].copy() for k in ['operators','charge','spin','q0','linear']}
        self.reference=FrozenNonlocalResponseReference.from_vertical_model(self.s,case['ctx'].kernel.partition,data,sha(model))
        self.context=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=self.s,spin_reference=self.reference)
        source=self.out/'coherent_common/ground/stationary.npz'
        if not source.exists():source=self.out/'coherent_continuation_models/t_1.00000000/ground/stationary.npz'
        with np.load(source) as z:cg=z['result_coefficients'][0].copy()
        cg,self.eg,_,_=refine.ground(self.context,cg);self.ground=GroundResponse(self.context,cg)
        self.energy=DensityEnergy(self.context);self.states=[];self.rows=[]
        rows=[dict(row,cohort=cohort) for row in json.loads((OUT/cohort/'manifest.json').read_text())['states'] if row['molecule']==name]
        for row in rows:
            folder=self.out/'coherent_states'/row['id'];report=json.loads((folder/'result.json').read_text());assert report['qualified']
            with np.load(folder/'final/final.npz') as z:c=z['coefficients'].copy()
            transition=target_transition(case,row);q=cg[:,:self.no].T@self.s.overlap@transition@self.s.overlap@cg[:,self.no:];q/=np.linalg.norm(q)
            c,mu,_,_,_=refine.roks(self.context,case['eri'],c,cg,q,report['multiplier'])
            self.states.append(ROKSResponse(self.context,case['eri'],c,self.ground,mu,q));self.rows.append(row)
        operator_path=self.destination/'physical_operators.npz'
        if operator_path.exists():
            with np.load(operator_path) as z:self.h=z['h'].copy();self.w=z['eri'].copy()
        else:
            inv=metric_power(self.s.overlap,-.5);mol=case['mol'];j,k=scf.hf.get_jk(mol,case['core_density'],hermi=1)
            physical=mol.intor('int1e_kin')+mol.intor('int1e_nuc')+j-.5*k
            self.h=inv@case['lift'].T@physical@case['lift']@inv
            self.w=np.einsum('pqrs,pi,qj,rk,sl->ijkl',case['eri'],inv,inv,inv,inv,optimize=True)
            np.savez_compressed(operator_path,h=self.h,eri=self.w)
        report=json.loads((DEST/'nosi_coherent'/cohort/name/'projected_result.json').read_text())
        self.roots={r['id']:r['root']-1 for r in report['predictions']};self.vertical={r['id']:r['gap_ev'] for r in report['predictions']}
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
        size=len(basis);metric=Jet.constant(np.zeros((size,size)),self.nr);hf=Jet.constant(np.zeros((size,size)),self.nr)
        for i in range(size):
            for j in range(i+1):
                overlap,value=csf_element(basis[i],basis[j],self.h,self.w)
                metric[i,j]=metric[j,i]=overlap;hf[i,j]=hf[j,i]=value
        h=Jet.constant(np.zeros((size,size)),self.nr)
        residual=[energy-hf[i,i]/metric[i,i] for i,energy in enumerate(diagonal)]
        for i in range(size):
            for j in range(size):h[i,j]=hf[i,j]+.5*metric[i,j]*(residual[i]+residual[j])
        return h,metric,diagonal[0]
    def hessian(self):
        values={};gradients={};hessians={row['id']:np.full((self.nr,self.nr),np.nan) for row in self.rows};ground_hessian=np.full((self.nr,self.nr),np.nan)
        progress=self.destination/'hessian_progress.npz';start=0
        if progress.exists():
            with np.load(progress) as z:
                assert str(z['source_sha256'])==sha(__file__),'Keep the previous checkpoint before changing the response implementation'
                start=int(z['completed']);ground_hessian=z['ground'].copy()
                for identity in hessians:
                    hessians[identity]=z[identity+'_hessian'].copy();gradients[identity]=z[identity+'_gradient'].copy();values[identity]=float(z[identity+'_energy'])
        for column in range(start,self.nr):
            direction=np.zeros(self.nr);direction[column]=1.;h,s,eg=self.matrices(direction);ground_hessian[:,column]=eg.ab
            for identity,root in self.roots.items():
                jet=projected_eigen_jet(h,s,root);gap=(float(jet.v)-float(eg.v))*EV
                assert abs(gap-self.vertical[identity])<1e-5,(identity,gap,self.vertical[identity])
                values[identity]=float(jet.v);gradients[identity]=jet.a;hessians[identity][:,column]=jet.ab
            np.savez_compressed(progress,completed=column+1,ground=ground_hessian,source_sha256=sha(__file__),
                **{identity+'_hessian':v for identity,v in hessians.items()},**{identity+'_gradient':v for identity,v in gradients.items()},**{identity+'_energy':v for identity,v in values.items()})
            print(self.case['src'].name,'analytic projected-NOSI Hessian',column+1,'/',self.nr,flush=True)
        result={}
        for identity,h in hessians.items():
            asym=float(np.max(abs(h-h.T)));assert asym<2e-6,(identity,asym)
            hh=.5*(h+h.T);translation=np.tile(np.eye(3),(self.nr//3,1));translation_error=float(np.max(abs(hh@translation)))
            assert translation_error<2e-6,(identity,translation_error)
            result[identity]=dict(energy_hartree=values[identity],gradient=gradients[identity],hessian=hh,raw_symmetry_error=asym,translation_error=translation_error)
        np.savez_compressed(self.destination/'nuclear_response.npz',ground_hessian=ground_hessian,**{f'{identity}_gradient':x['gradient'] for identity,x in result.items()},**{f'{identity}_hessian':x['hessian'] for identity,x in result.items()})
        return result
