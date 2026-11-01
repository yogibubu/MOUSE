"""Prepared-only joint parent continuation with analytic overlap constraints."""
from predict import *
from joint_orthogonal import solve_joint
from matrix_gfn2.multicsf import MultiCSFObjective
from native_batch import install_native_batch

def predict_joint(cohort,name):
    out=DEST/cohort/name;assert (out/'calibration_complete.json').exists();case=prepare(cohort,name)
    s=case['s'];no=case['no'];install_native_batch(case,out)
    case['calc'].evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC forbidden in prepared joint prediction'))
    scf.hf.kernel=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('DFT SCF forbidden in prepared joint prediction'))
    rows=[dict(r,cohort=cohort) for r in json.loads((OUT/cohort/'manifest.json').read_text())['states'] if r['molecule']==name]
    for row in rows:
        dest=out/'joint_states'/row['id'];dest.mkdir(parents=True,exist_ok=True)
        if (dest/'result.json').exists():
            if json.loads((dest/'result.json').read_text())['qualified']:continue
            (dest/'result.json').replace(dest/'prior_numerical_attempt.json')
        transition=target_transition(case,row);point=[];grounds=[]
        for i,profile in enumerate(['F','Q']):
            earlier=Path('/home/enzo/MOUSE-transferability-20261101/stationary_analytic/orthogonal')/cohort/name/row['id']/f'parent_{i}/final.npz'
            with np.load(earlier) as a:point.append(a['coefficients'].copy())
            oldpath=case['src']/['full_response','qualified_spectral_response'][i]/'anchor/model.npz'
            with np.load(oldpath) as f:data={k:f[k].copy() for k in ['operators','charge','spin','q0','linear']}
            oldref=FrozenNonlocalResponseReference.from_vertical_model(s,case['ctx'].kernel.partition,data,sha(oldpath))
            grounds.append(ground(case['src']/f'grounds/{i}',case['calc'],case['xyz'],s,oldref))
        current=0.;step=.125;accepted=dest/'accepted.npz'
        if accepted.exists():
            with np.load(accepted) as a:current=float(a['t']);point=[c.copy() for c in a['coefficients']]
        trace=json.loads((dest/'continuation.json').read_text()) if (dest/'continuation.json').exists() else []
        attempts=0
        while current<1.-1e-12 and attempts<128:
            attempts+=1;target=min(1.,current+step);objects=[];gg=[]
            for profile,g in zip(['F','Q'],grounds):
                response,g,ctx=response_and_ground(case,out,profile,target,g);cg=g.coefficients[0]
                q=cg[:,:no].T@s.overlap@transition@s.overlap@cg[:,no:];q/=np.linalg.norm(q)
                objects.append(MultiCSFObjective(ctx,case['eri'],no,1,cg,q));gg.append(g)
            trial_dir=dest/f'from_{current:.8f}_to_{target:.8f}'
            result,trial=solve_joint(objects,gg,point,trial_dir)
            if not result['qualified']:
                trace.append(dict(t_from=current,t_target=target,status='subdivide',kkt=result['kkt_max'],fidelity=result['state_fidelities']))
                write(dest/'continuation.json',trace);step*=.5
                if step<1/256:break
                continue
            current=target;point=trial;grounds=gg
            np.savez_compressed(dest/'accepted.tmp.npz',t=current,coefficients=np.array(point));(dest/'accepted.tmp.npz').replace(accepted)
            trace.append(dict(t=current,status='accepted',kkt=result['kkt_max'],fidelity=result['state_fidelities']))
            write(dest/'continuation.json',trace);print(name,row['id'],'joint t=',current,flush=True)
            if result['iterations']<=8:step=min(.125,1.5*step)
        report=dict(id=row['id'],molecule=name,family=row['family'],qualified=current>=1.-1e-12,reached_t=current,
            continuation=trace,energy_labels_used=False,weights=[.5,.5],lambda_exchange=.9,offset_ev=0,
            source_sha256=sha(__file__),protocol_sha256=sha(DEST/'joint_protocol.json'),host=HOST,threads=2)
        if report['qualified']:
            report.update(kkt_max=result['kkt_max'],ground_overlaps=result['ground_overlaps'],state_fidelities=result['state_fidelities'],active_frame_drift=result['active_frame_drift'],
                parent_gaps_ev=[(e-g.energy_hartree)*EV for e,g in zip(result['parent_energies'],grounds)])
            report['excitation_ev']=.5*sum(report['parent_gaps_ev'])
        write(dest/'result.json',report)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('cohort');a.add_argument('name');args=a.parse_args()
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():predict_joint(args.cohort,args.name)
