"""Nonlinear DFT remainder diagnostic on completed spin-polarized OO states."""
from pathlib import Path
import sys,json,socket
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
ROOT=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-response-consistent-nosi-20261101');HERE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-pi-pi-transfer-repair-20261101');CAM=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-cam-response-20261101')
sys.path.insert(0,str(ROOT))
from basis import *
from anchor import CompactDFT

def run(item):
 cohort,name=item['cohort'],item['molecule'];out=HERE/'nonlinear_xc'/cohort/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 d=prepare_basis(cohort,name);case=d['case'];no=d['no'];n=len(d['cg']);occ=np.zeros((2,n));occ[:,:no]=1;p=d['cg'][:,:no]@d['cg'][:,:no].T;densities=[];labels=[]
 for row in d['rows']:
  with np.load(d['source']/'coherent_states'/row['id']/'final/final.npz') as z:c=z['coefficients'].copy()
  mix=occ.copy();mix[0,0]=0;mix[0,no]=1;trip=np.array([np.maximum(*mix),np.minimum(*mix)]);double=occ.copy();double[:,0]=0;double[:,no]=1
  for label,occupation in [('mixed',mix),('triplet',trip),('double',double)]:densities.append(np.array([(c*x)@c.T for x in occupation]));labels.append(row['id']+'_'+label)
 functional=CompactDFT(case,CAM/cohort/name/'grid');remainder=np.zeros(len(densities));curvature=np.zeros(len(densities));linear=np.zeros(len(densities));exact=np.zeros(len(densities))
 for phi,core,weight in functional.chunks:
  rho0=.5*core+functional.rho(phi,p);rho0=np.array([rho0,rho0]);exc,vxc,fxc,*_=functional.mf._numint.eval_xc_eff('CAM-B3LYP',rho0,deriv=2,xctype='GGA');e0=weight@((rho0[0,0]+rho0[1,0])*exc)
  for i,dm in enumerate(densities):
   rho=np.array([.5*core+functional.rho(phi,x) for x in dm]);delta=rho-rho0;ex=functional.mf._numint.eval_xc_eff('CAM-B3LYP',rho,deriv=0,xctype='GGA')[0];e=weight@((rho[0,0]+rho[1,0])*ex);de=np.einsum('aig,aig,g->',vxc,delta,weight,optimize=True);d2e=.5*np.einsum('aibjg,aig,bjg,g->',fxc,delta,delta,weight,optimize=True)
   remainder[i]+=e-e0-de-d2e;exact[i]+=e-e0;linear[i]+=de;curvature[i]+=d2e
 rows=[]
 for i,row in enumerate(d['rows']):rows.append(dict(id=row['id'],xc_roks_remainder_ev=(2*remainder[3*i]-remainder[3*i+1])*EV,xc_double_remainder_ev=remainder[3*i+2]*EV))
 write(out/'result.json',dict(rows=rows,labels=labels,remainder_ev=remainder*EV,exact_xc_change_ev=exact*EV,linear_ev=linear*EV,quadratic_ev=curvature*EV));print(name,rows,flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--molecule');args=p.parse_args()
 with threadpool_limits():
  for item in json.loads((HERE/'cases.json').read_text()):
   if args.molecule is None or item['molecule']==args.molecule:run(item)
