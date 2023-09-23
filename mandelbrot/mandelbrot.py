#!/usr/bin/env python
"""
mandelbrot.py 
==============

"""
import numpy as np
SIZE = np.array([1280, 720])
import matplotlib.pyplot as plt

def read_npy(path, d):
    path = os.path.expandvars(path)
    a = np.load(path)
    if not d is None:
        txtpath = path.replace(".npy","_meta.txt")
        lines = open(txtpath).read().splitlines()
        for line in lines:
            key, val = line.split(":")
            d[key] = val
        pass 
    pass
    return a

if __name__ == '__main__':

    d = dict()
    a = read_npy("$FOLD/a.npy", d)

    d["CMAP"] = os.environ.get("CMAP", "prism")
    cmap = getattr(plt.cm, d["CMAP"], None)
    d["extent"] = list(map(float,(d["xmin"], d["xmax"], d["ymin"], d["ymax"] )))
    d["ami"] = a.min()
    d["amx"] = a.max()

    label = "mandelbrot.sh : CMAP %(CMAP)s FOCUS %(FOCUS)s MZZ %(MZZ)s" 
    label += " MIT %(MIT)s extent %(extent)s ami %(ami)d amx %(amx)d "
    print(label % d)

    fig, ax = plt.subplots(figsize=SIZE/100.)  
    fig.suptitle(label % d)
    ax.imshow(a, extent=d["extent"], cmap=cmap) 
    fig.show()

