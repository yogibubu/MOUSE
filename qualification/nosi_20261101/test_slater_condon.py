import socket
assert socket.getfqdn()=='oracle.sns.it'
import numpy as np
from scipy.linalg import expm
from slater_condon import *

def setup():
    rng=np.random.default_rng(7);h=rng.normal(size=(4,4));h=.5*(h+h.T)
    v=rng.normal(size=(4,4,4,4));v=.5*(v+v.swapaxes(0,1));v=.5*(v+v.swapaxes(2,3));v=.5*(v+v.transpose(2,3,0,1))
    return rng,h,v

def test_diagonal_and_singular_double():
    rng,h,v=setup();c=np.eye(4);a=(c[:,:2],c[:,:2]);p=2*c[:,:2]@c[:,:2].T
    s,e=element(a,a,h,v)
    expected=np.sum(p*h)+.5*np.einsum('pqrs,pq,rs',v,p,p)-.25*np.einsum('prqs,pq,rs',v,p,p)
    assert abs(s-1)<1e-12 and abs(e-expected)<1e-12
    b=(c[:,[0,2]],c[:,[0,2]])
    s,e=element(a,b,h,v)
    assert abs(s)<1e-12 and abs(e-v[1,2,1,2])<1e-12

def test_hermiticity_and_occupied_gauge():
    rng,h,v=setup();q,_=np.linalg.qr(rng.normal(size=(4,4)));r,_=np.linalg.qr(rng.normal(size=(4,4)))
    a=(q[:,:2],q[:,1:3]);b=(r[:,:2],r[:,2:4])
    assert np.max(abs(np.array(element(a,b,h,v))-element(b,a,h,v)))<1e-12
    rot=np.array([[np.cos(.4),-np.sin(.4)],[np.sin(.4),np.cos(.4)]])
    assert np.max(abs(np.array(element(a,b,h,v))-element((a[0]@rot,a[1]),b,h,v)))<1e-12

def test_singlet_sign_and_hf_limit():
    rng,h,v=setup();c=np.eye(4)
    # Fixed alpha-block/beta-block determinant order: positive spatial singlet.
    x=(c[:,[0,2]],c[:,[0,1]]);y=(c[:,[0,1]],c[:,[0,2]])
    singlet=[(2**-.5,x),(2**-.5,y)]
    _,em=element(x,x,h,v);_,et=element((c[:,[0,1,2]],c[:,[0]]),(c[:,[0,1,2]],c[:,[0]]),h,v)
    _,es=csf_element(singlet,singlet,h,v)
    assert abs(es-(2*em-et))<1e-12
