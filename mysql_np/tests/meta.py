import os, numpy as np

npzs = lambda dir:filter(lambda _:_.endswith(".npz") , os.listdir(dir))

dir = ".npz"
for npz in npzs(dir):
    print "\n ========= %s =========== " % npz
    d = np.load(os.path.join(dir,npz)) 
    if not "meta" in d:continue
    if not "scan" in d:continue

    _meta = d["meta"]
    meta = dict(zip(_meta.dtype.names, _meta[0]))
    sc = d["scan"]

    print " meta : %s " % repr(meta)
    print "   sc : %s " % repr(sc)
    pass




