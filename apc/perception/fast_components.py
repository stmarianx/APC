"""Numpy-batched row-run extraction with incumbent four-connectivity/order."""
import numpy as np


def fast_components(mask):
    mask = np.asarray(mask,dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional mask")
    padded = np.pad(mask,((0,0),(1,1)),constant_values=False)
    ys,xs = np.nonzero(padded[:,1:] != padded[:,:-1])
    parents,runs,previous = [],[],[]
    previous_y = -2
    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    current = []
    for y,left,right in zip(ys[::2],xs[::2],xs[1::2]):
        y,left,right = int(y),int(left),int(right)
        if y != previous_y:
            previous = current if y == previous_y+1 else []
            current = []
            previous_y = y
        index = len(parents)
        parents.append(index)
        for a,b,other in previous:
            if b <= left:
                continue
            if a >= right:
                break
            one,two = root(index),root(other)
            parents[max(one,two)] = min(one,two)
        current.append((left,right,index))
        runs.append((left,right,y,index))
    groups = {}
    for left,right,y,index in runs:
        key = root(index)
        if key not in groups:
            groups[key] = [left,y,right,y+1,right-left]
        else:
            item = groups[key]
            item[0],item[1],item[2],item[3] = min(item[0],left),min(item[1],y),max(item[2],right),max(item[3],y+1)
            item[4] += right-left
    return [tuple(groups[key]) for key in sorted(groups)]
