"""Frozen-ground-space check of the rotation-covariant pair closure."""
import state_average_diagnostic as driver
from state_average_diagnostic import *
from covariant_pair import project_pair_matrix,pair_rotation

def probe(context,no,ground,q,out):
    out.mkdir(parents=True,exist_ok=True)
    u,sv,vh=np.linalg.svd(q,full_matrices=True)
    c=np.column_stack([ground[:,:no]@u,ground[:,no:]@vh.T]);matrices={}
    for nodes in [8,16,32]:matrices[nodes]=project_pair_matrix(context,c,no,nodes=nodes)
    values,z=np.linalg.eigh(matrices[32]);angle=.371
    cc=c.copy();cc[:,0]=np.cos(angle)*c[:,0]-np.sin(angle)*c[:,no];cc[:,no]=np.sin(angle)*c[:,0]+np.cos(angle)*c[:,no]
    rotated=project_pair_matrix(context,cc,no,nodes=32);d=pair_rotation(angle)
    covariance=float(np.max(abs(rotated-d.T@matrices[32]@d))*EV)
    convergence=float(np.max(abs(matrices[32]-matrices[16]))*EV)
    result=dict(gap_ev=float((values[1]-values[0])*EV),qualified=bool(covariance<1e-7 and convergence<1e-7),
        target_assignment_qualified=bool(z[1,1]**2>.5),open_singlet_weight=float(z[1,1]**2),
        eigenvalues=values,ci=z,matrix=matrices[32],nodes=32,
        matrix_convergence_16_to_32_ev=convergence,covariance_error_ev=covariance,
        energy_labels_used=False,source_sha256=sha(__file__),closure_source_sha256=sha(Path(__file__).parent/'covariant_pair.py'),
        scope='vertical group-projected closure at ground NTO orbitals, not an optimized molecular surface')
    write(out/'projection_certificate.json',result);return result

if __name__=='__main__':
    driver.solve=probe;driver.DEST=DEST/'covariant_vertical'
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        for cohort,name,state in [('independent','Criegee_Intermediate','I002'),('independent_challenge','Furofuran','C002')]:
            driver.run(cohort,name,state)
