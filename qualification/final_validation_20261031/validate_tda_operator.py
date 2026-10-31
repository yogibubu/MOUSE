"""Verify the new helper against the existing independently qualified TDA matrix."""
from common import *
from tda_operator import complete_tda_A
import fcntl
lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
with threadpool_limits():
 d=Path('/home/enzo/MOUSE-state-following-20261031/reduced_validation/tda/Furan/0/spectrum.npz')
 with np.load(d) as a:static=a['static_matrix'].copy();reference=a['matrix'].copy();c=a['coefficients'].copy()
 with np.load(P/'results/full_response/Furan/projected_eri.npz') as a:eri=a['eri'].copy()
 no=MultiCSFModel(W/'models/Furan/B022/model.json').no;matrix,correction=complete_tda_A(static,eri,c,no);error=float(np.max(abs(matrix-reference)));assert error<1e-10
 certificate=Path('/home/enzo/MOUSE-state-following-20261031/tda_exchange_validation.json');source=Path('/home/enzo/MOUSE-state-following-20261031/verify_tda_exchange.py')
 write(OUT/'true_tda_operator_validation.json',dict(status='passed',full_matrix_error_hartree=error,correction_max_hartree=float(np.max(abs(correction))),existing_independent_pyscf_certificate=json.loads(certificate.read_text()),existing_certificate_sha256=sha(certificate),independent_check_source_sha256=sha(source),helper_source_sha256=sha(Path(__file__).parent/'tda_operator.py'),scope='exact reproduction of the stored corrected GFN2-orbital TDA matrix; existing independent PySCF A validation retained'))
 print('TRUE TDA OPERATOR PASS',error,flush=True)
