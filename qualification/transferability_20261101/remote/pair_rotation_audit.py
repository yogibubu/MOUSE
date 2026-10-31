"""Internal-active-orbital gauge audit: a full CI spectrum must be invariant."""
import state_average_diagnostic as driver
from state_average_diagnostic import *
from pair_cas import evaluate_pair_cas

def audit(context,no,ground,q,out):
    out.mkdir(parents=True,exist_ok=True)
    u,sv,vh=np.linalg.svd(q,full_matrices=True)
    c0=np.column_stack([ground[:,:no]@u,ground[:,no:]@vh.T]);rows=[]
    for angle in np.linspace(0.,np.pi,13):
        c=c0.copy();co=np.cos(angle);si=np.sin(angle)
        c[:,0]=co*c0[:,0]-si*c0[:,no];c[:,no]=si*c0[:,0]+co*c0[:,no]
        result=evaluate_pair_cas(context,c,no,hole=0,particle=no)
        rows.append(dict(angle_radians=float(angle),energies_hartree=result.energies))
    energies=np.array([r['energies_hartree'] for r in rows])
    report=dict(internal_rotation_only=True,active_projector_unchanged=True,core_projector_unchanged=True,
        maximum_spectrum_drift_ev=float(np.max(abs(energies-energies[0]))*EV),
        lowest_two_average_span_ev=float(np.ptp(.5*(energies[:,0]+energies[:,1]))*EV),rows=rows,
        energy_labels_used=False,host=HOST,source_sha256=sha(__file__))
    write(out/'rotation_audit.json',report)
    return dict(gap_ev=float((energies[0,1]-energies[0,0])*EV),qualified=False,
        interpretation='Gauge audit only; this return value is not a candidate prediction')

if __name__=='__main__':
    driver.solve=audit;driver.DEST=DEST/'rotation_audit'
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        for cohort,name,state in [('independent','Criegee_Intermediate','I002'),('independent_challenge','Furofuran','C002')]:
            driver.run(cohort,name,state)
