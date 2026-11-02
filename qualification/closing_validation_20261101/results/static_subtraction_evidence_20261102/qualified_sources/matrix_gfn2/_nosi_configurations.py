"""Frozen, explicit spin-singlet configurations for extended OO-CID spaces."""
import re
import numpy as np
from ._nosi_spin import spin_vectors,ASSIGNMENTS


def from_labels(labels,no):
    records=[]
    for label in labels:
        match=re.fullmatch(r'single_h(\d+)_p(\d+)',label)
        if match:
            hole,particle=map(int,match.groups())
            if not 0<=hole<no<=particle:
                raise ValueError('invalid single-excitation orbital indices')
            records.append(dict(kind='open',closed=[i for i in range(no) if i!=hole],open=[hole,particle]))
            continue
        match=re.fullmatch(r'pair_h(\d+)_p(\d+)',label)
        if match:
            hole,particle=map(int,match.groups())
            records.append(dict(kind='closed',closed=[i for i in range(no) if i!=hole]+[particle]))
            continue
        match=re.fullmatch(r'(open|four)_h(\d+)_(\d+)_p(\d+)_(\d+)(?:_([01]))?',label)
        if not match:raise ValueError('invalid extended CID label')
        kind,hi,hj,a,b,component=match.groups();hi,hj,a,b=map(int,(hi,hj,a,b))
        core=[i for i in range(no) if i not in (hi,hj)]
        if kind=='four':records.append(dict(kind='four',closed=core,open=[hi,hj,a,b],component=int(component)))
        elif hi==hj:records.append(dict(kind='open',closed=core,open=[a,b]))
        elif a==b:records.append(dict(kind='open',closed=core+[a],open=[hi,hj]))
        else:raise ValueError('invalid open-shell CID label')
    return records


def validate(records,no,n):
    for i,record in enumerate(records):
        kind=record.get('kind');closed=record.get('closed',[]);opened=record.get('open',[])
        if kind not in ('closed','open','four') or set(record)-{'kind','closed','open','component'}:raise ValueError('invalid frozen CID configuration')
        if any(type(j) is not int or not 0<=j<n for j in closed+opened) or len(set(closed+opened))!=len(closed+opened):raise ValueError('invalid CID orbital indices')
        if 2*len(closed)+len(opened)!=2*no or len(opened)!={'closed':0,'open':2,'four':4}[kind]:raise ValueError('invalid CID occupation')
        if kind=='four':
            component=record.get('component')
            if component not in (0,1):raise ValueError('invalid singlet recoupling component')
            partner=i+1 if component==0 else i-1
            if not 0<=partner<len(records):raise ValueError('incomplete four-spin block')
            other=records[partner]
            if other.get('kind')!='four' or other.get('component')!=1-component or other.get('closed')!=closed or other.get('open')!=opened:raise ValueError('inconsistent four-spin pair')
    return records


def basis(y,records):
    result=[];vectors,phases=spin_vectors()
    for record in records:
        core=record['closed'];opened=record.get('open',[]);kind=record['kind']
        if kind=='closed':result.append([(1.,(y[:,core],y[:,core]))])
        elif kind=='open':
            a=y[:,core+[opened[0]]];b=y[:,core+[opened[1]]]
            result.append([(2**-.5,(a,b)),(2**-.5,(b,a))])
        else:
            csf=[]
            for row,alpha in enumerate(ASSIGNMENTS):
                weight=vectors[row,record['component']]*phases[row]
                if weight:csf.append((float(weight),(y[:,core+[opened[j] for j in alpha]],y[:,core+[opened[j] for j in range(4) if j not in alpha]])))
            result.append(csf)
    return result


def common_core_and_active(records):
    core=set(records[0]['closed']);occupied=set()
    for record in records:
        core.intersection_update(record['closed']);occupied.update(record['closed']);occupied.update(record.get('open',[]))
    return sorted(core),sorted(occupied-core)


def frontier_rank_two(no, frontier):
    """Complete singlet rank <= 2 records in the historical lexical order.

    Enumerate occupied holes and virtual particles directly, avoiding the
    exponentially large space of occupations that would subsequently be rejected.
    """
    from itertools import combinations
    if not 2 <= frontier <= no:
        raise ValueError('invalid occupied frontier')
    ground = set(range(no))
    holes = range(no-frontier, no)
    particles = range(no, no+frontier)
    closed = [dict(kind='closed', closed=list(range(no)))]
    opened = []
    four = []
    for h in holes:
        core = sorted(ground-{h})
        for a in particles:
            closed.append(dict(kind='closed', closed=sorted(ground-{h}|{a})))
            opened.append(dict(kind='open', closed=core, open=[h,a]))
        for a,b in combinations(particles,2):
            opened.append(dict(kind='open', closed=core, open=[a,b]))
    for h,j in combinations(holes,2):
        core = ground-{h,j}
        for a in particles:
            opened.append(dict(kind='open', closed=sorted(core|{a}), open=[h,j]))
        for a,b in combinations(particles,2):
            for component in range(2):
                four.append(dict(kind='four', closed=sorted(core), open=[h,j,a,b], component=component))
    key = lambda r: (tuple(r['closed']),tuple(r.get('open',[])),r.get('component',0))
    return sorted(closed,key=key)+sorted(opened,key=key)+sorted(four,key=key)
