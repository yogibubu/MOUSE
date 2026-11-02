"""Reuse NumPy's greedy contraction program without repeating path searches.

Only index/shape metadata are cached. Arithmetic and contraction ordering are
NumPy's optimize=True program; no numerical tensor or approximation is cached.
Unsupported planner versions and nondefault options use NumPy directly.
"""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=512)
def _program(indices,shapes):
    operands=[np.broadcast_to(np.array(0.),shape) for shape in shapes]
    try:
        _,contractions=np.einsum_path(indices,*operands,optimize='greedy',einsum_call=True)
    except TypeError:
        return None
    program=[]
    for positions,removed,expression,_remaining,blas in contractions:
        axes=None;reorder=None
        if blas:
            inputs,result=expression.split('->');left,right=inputs.split(',')
            axes=(tuple(left.index(i) for i in sorted(removed)),tuple(right.index(i) for i in sorted(removed)))
            product=left+right
            for i in removed:product=product.replace(i,'')
            if product!=result:reorder=product+'->'+result
        program.append((tuple(positions),expression,axes,reorder))
    return tuple(program)


def einsum(indices,*operands,optimize=False,**kwargs):
    if optimize is not True or kwargs:
        return np.einsum(indices,*operands,optimize=optimize,**kwargs)
    program=_program(indices,tuple(np.shape(operand) for operand in operands))
    if program is None:return np.einsum(indices,*operands,optimize=True)
    work=list(operands)
    for positions,expression,axes,reorder in program:
        values=[work.pop(position) for position in positions]
        if axes is None:result=np.einsum(expression,*values,optimize=False)
        else:
            result=np.tensordot(*values,axes=axes)
            if reorder is not None:result=np.einsum(reorder,result,optimize=False)
        work.append(result)
    return np.asanyarray(work[0],order='K')
