"""Independent checks that observable and Hamiltonian CSFs share determinant phases."""
import pytest
from run_pigment_cid import *
from matrix_gfn2._nosi_nuclear import opened as production_opened
from matrix_gfn2._nosi_mixed import Jet

@pytest.mark.parametrize('no',[4,5])
def test_observer_matches_production_phase(no):
 y=np.linalg.qr(np.random.default_rng(no).normal(size=(2*no,2*no)))[0]
 expected=[(w,(a.v,b.v)) for w,(a,b) in production_opened(Jet.constant(y,0),no)]
 assert abs(overlap(opened(y,no),expected)-1)<1e-12
 for h in range(no):
  p=no;order=[h]+[j for j in range(no) if j!=h]+[p]+[j for j in range(no,2*no) if j!=p]
  record=dict(kind='open',closed=[j for j in range(no) if j!=h],open=[h,p])
  assert abs(overlap(opened(y[:,order],no),configuration_basis(y,[record])[0])-(-1)**(no-1))<1e-12

def test_observable_basis_matches_archived_physical_metric():
 folder=HERE/'pigment_cid_expansion/Beta_carotene/screening_10/frontier_8';report=json.loads((folder/'result.json').read_text());_,state=load_state_only('Beta_carotene');no=state.electron_count//2;n=len(state.overlap)
 v,u=eigh(state.overlap);sr=(u*v**.5)@u.T;frame=TransportedOperators(state,state.coordinates_bohr,np.zeros((0,n,n))).u;y=frame.T@sr@state.coefficients;ys=[]
 for ordinal in report['retained_ROKS_ordinals']:
  with np.load(HERE/f'pigment_roks/Beta_carotene/state_{ordinal}/checkpoint.npz') as z:ys.append(frame.T@sr@z['coefficients'])
 ids=[0,1,2]+[3+j for j in [0,1,2,16,80,100,500,1000]];basis=scalar_basis(y,ys,[report['auxiliary_records'][j-3] for j in ids[3:]],no)
 with np.load(folder/'matrices.npz') as z:metric=z['s'][np.ix_(ids,ids)]
 actual=np.array([[overlap(a,b) for b in basis] for a in basis]);assert np.max(abs(actual-metric))<1e-10
