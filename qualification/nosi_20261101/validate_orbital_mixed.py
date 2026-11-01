"""Criegee ground and both ROKS roots: analytic mixed response validation."""
from pathlib import Path
import sys
sys.path.insert(0,'/home/enzo/MOUSE-m06-response-20261101')
from predict import *
from native_batch import install_native_batch
from orbital_mixed import GroundResponse,ROKSResponse,densities
import refine

HERE=Path('/home/enzo/MOUSE-nosi-20261101/orbital_validation')

def run():
    cohort='independent';name='Criegee_Intermediate';case=prepare(cohort,name);out=DEST/cohort/name;s=case['s'];no=case['no'];n=len(s.overlap)
    install_native_batch(case,out)
    model=out/'coherent_common/model.npz'
    with np.load(model) as z:data={k:z[k].copy() for k in ['operators','charge','spin','q0','linear']}
    reference=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,data,sha(model))
    context=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=s,spin_reference=reference)
    with np.load(out/'coherent_continuation_models/t_1.00000000/ground/stationary.npz') as z:cg=z['result_coefficients'][0].copy()
    cg,eg,g_res,gi=refine.ground(context,cg);ground=GroundResponse(context,cg)
    direction=np.zeros(s.coordinates_bohr.size);direction[3]=1.;metric=ground.engine.geometry(direction)[1]
    gs=ground.coefficient(direction);responses=[];states=[];results=[]
    rows=[dict(row,cohort=cohort) for row in json.loads((OUT/cohort/'manifest.json').read_text())['states'] if row['molecule']==name]
    for row in rows:
        folder=out/'coherent_states'/row['id'];report=json.loads((folder/'result.json').read_text())
        with np.load(folder/'final/final.npz') as z:c=z['coefficients'].copy()
        transition=target_transition(case,row);q=cg[:,:no].T@s.overlap@transition@s.overlap@cg[:,no:];q/=np.linalg.norm(q)
        c,mu,e,res,it=refine.roks(context,case['eri'],c,cg,q,report['multiplier'])
        response=ROKSResponse(context,case['eri'],c,ground,mu,q);cj=response.coefficient(direction)
        residual=response.residual(cj,gs,metric,response.cache[direction.tobytes()][1],direction)
        maximum=float(np.max(abs(residual.ab)));assert maximum<2e-7,maximum
        normalization=cj.T@metric@cj
        assert max(np.max(abs(normalization.a)),np.max(abs(normalization.ab)))<1e-9
        dm=densities(cj,response.occupations);responses.append(response);states.append((c,mu,q))
        results.append(dict(id=row['id'],primal_residual=res,first_response_residual=response.first_residual,
            mixed_KKT_residual=maximum,minimum_KKT_singular_value=response.minimum_singular_value,center_refinement_iterations=it,finite_difference=[]))
        np.savez_compressed(HERE/(row['id']+'_response.npz'),coefficients=c,first_coefficients=cj.a,mixed_coefficients=cj.ab,multiplier=mu,first_multiplier=response.first_multiplier,
            density=np.array([p.v for p in dm]),first_density=np.array([p.a for p in dm]),mixed_density=np.array([p.ab for p in dm]))
    for eps in [1e-3,5e-4]:
        points=[]
        for sign in [-1,1]:
            destination=HERE/f'eps_{eps:.8f}_sign_{sign}';destination.mkdir(parents=True,exist_ok=True)
            checkpoint=destination/'roots.npz'
            if checkpoint.exists():
                with np.load(checkpoint) as z:points.append(z['densities'].copy())
                continue
            xyz=case['xyz']+sign*eps*direction.reshape(-1,3)/ANGSTROM_TO_BOHR
            calc=calculator(s);calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC forbidden in independent validation'))
            ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=reference)
            gc,_,gres,_=refine.ground(ctx,cg)
            densities_at_point=[];residuals=[]
            for (c,mu,q),response in zip(states,responses):
                cc,mm,ee,r,it=refine.roks(ctx,case['eri'],c,gc,q,mu)
                densities_at_point.append(np.array([(cc*x)@cc.T for x in response.occupations]));residuals.append(r)
            np.savez_compressed(checkpoint,densities=densities_at_point,ground_residual=gres,roks_residuals=residuals)
            points.append(np.array(densities_at_point))
        for index,response in enumerate(responses):
            cj=response.coefficient(direction);dm=densities(cj,response.occupations)
            first=np.array([direction@p.a.reshape(p.n,-1) for p in dm]).reshape(2,n,n)
            second=np.array([direction@p.ab.reshape(p.n,-1) for p in dm]).reshape(2,n,n)
            center=np.array([p.v for p in dm]);gfd=(points[1][index]-points[0][index])/(2*eps)
            hfd=(points[1][index]-2*center+points[0][index])/eps**2
            ge=float(np.max(abs(gfd-first)));he=float(np.max(abs(hfd-second)))
            assert ge<3e-6,(index,eps,ge)
            assert he<3e-5,(index,eps,he)
            results[index]['finite_difference'].append(dict(epsilon_bohr=eps,first_density_error=ge,second_density_error=he))
    write(HERE/'result.json',dict(host=HOST,ground_primal=g_res,ground_first_response=ground.first_residual,states=results,
        implementation='analytic mixed KKT including moving ground, metric, multiplier and overlap constraint',source_sha256=sha(__file__),model_sha256=sha(model)))
    print(results,flush=True)

if __name__=='__main__':
    HERE.mkdir(parents=True,exist_ok=True)
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():run()
