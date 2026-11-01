"""Exact semilocal XC remainder on retained Oracle quadrature checkpoints."""
from pathlib import Path
import sys
sys.path.insert(0,'/home/enzo/MOUSE-stationary-anchor-20261101')
from anchor import *

HERE=Path('/home/enzo/MOUSE-xc-remainder-20261101')
ROOTS={'PBE0':Path('/home/enzo/MOUSE-stationary-anchor-20261101'),'M06-2X':Path('/home/enzo/MOUSE-m06-response-20261101')}

class Remainder:
    def __init__(self,source,xc):
        self.xc=xc;self.ni=dft.numint.NumInt();self.blocks=[];self.type='GGA' if xc=='PBE0' else 'MGGA'
        with np.load(source/'embedded_ground.npz') as z:p=z['density'].copy()
        for path in sorted((source/'grid').glob('grid_*.npz')):
            with np.load(path) as z:phi=z['phi'].copy();core=z['rho_core'].copy();weight=z['weight'].copy()
            rho0=np.array([.5*(core+self.rho(phi,p))]*2)
            exc,v,k=self.ni.eval_xc_eff(xc,rho0,deriv=2,xctype=self.type)[:3]
            e=float(np.sum(weight*rho0[:,0].sum(0)*exc))
            self.blocks.append((phi,.5*core,weight,rho0,e,v,k))
    def rho(self,phi,p):
        d=4 if self.type=='GGA' else 5;rho=np.empty((d,phi.shape[1]))
        rho[0]=np.einsum('gi,ij,gj->g',phi[0],p,phi[0],optimize=True)
        for k in range(1,4):rho[k]=2*np.einsum('gi,ij,gj->g',phi[k],p,phi[0],optimize=True)
        if d==5:rho[4]=.5*sum(np.einsum('gi,ij,gj->g',phi[k],p,phi[k],optimize=True) for k in range(1,4))
        return rho
    def energy(self,dm):
        total=0.
        for phi,core,w,rho0,e0,v0,k0 in self.blocks:
            rho=np.array([core+self.rho(phi,p) for p in dm]);dr=rho-rho0
            exc=self.ni.eval_xc_eff(self.xc,rho,deriv=0,xctype=self.type)[0]
            exact=float(np.sum(w*rho[:,0].sum(0)*exc))
            first=np.einsum('sig,sig,g->',v0,dr,w,optimize=True)
            second=.5*np.einsum('sitjg,sig,tjg,g->',k0,dr,dr,w,optimize=True)
            total+=exact-e0-first-second
        return float(total)

def run(xc,cohort,name):
    output=HERE/xc/cohort/name/'result.json'
    if output.exists():return
    source=ROOTS[xc]/cohort/name;case=prepare(cohort,name);s=case['s'];no=case['no'];n=len(s.overlap)
    remainder=Remainder(source,xc)
    with np.load(source/'embedded_ground.npz') as z:p=z['density'].copy()
    zero=remainder.energy(np.array([p,p])*.5);assert abs(zero)<1e-10
    rows=[row for row in json.loads((OUT/cohort/'manifest.json').read_text())['states'] if row['molecule']==name];predictions=[]
    for row in rows:
        if xc=='PBE0':
            state=source/'common_states'/row['id'];result=json.loads((state/'result.json').read_text())
            checkpoint=state/('stationary_from_F' if result.get('warm_start_checkpoint') else 'stationary')/'final.npz'
        else:
            state=source/'continued_states'/row['id'];result=json.loads((state/'result.json').read_text());checkpoint=state/'final/final.npz'
        if not result['qualified']:predictions.append(dict(id=row['id'],qualified=False));continue
        with np.load(checkpoint) as z:c=z['coefficients'].copy()
        occ=np.zeros((2,n));occ[:,:no]=1;occ[0,0]=0;occ[0,no]=1
        triplet=np.array([np.maximum(*occ),np.minimum(*occ)])
        values=[]
        for o in [occ,triplet]:values.append(remainder.energy(np.array([(c*x)@c.T for x in o])))
        change=(2*values[0]-values[1])*EV
        predictions.append(dict(id=row['id'],frozen_orbital_only=True,original_gap_ev=result['excitation_ev'],
            mixed_remainder_ev=values[0]*EV,triplet_remainder_ev=values[1]*EV,roks_remainder_ev=change,
            frozen_corrected_gap_ev=result['excitation_ev']+change,checkpoint_sha256=sha(checkpoint)))
    write(output,dict(xc=xc,host=HOST,zero_center_remainder=zero,rows=predictions,energy_labels_used=False,
        scope='frozen-orbital diagnostic, not stationary predictions; requires quadrature and is not a production semiempirical implementation',
        source_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json')))
    print(xc,name,predictions,flush=True)

if __name__=='__main__':
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        for xc in ['PBE0','M06-2X']:
            for cohort,name in [('independent','Criegee_Intermediate'),('independent_challenge','Furofuran')]:run(xc,cohort,name)
