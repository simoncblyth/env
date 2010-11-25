"""
   Use ipython to get a chance to see the plot ...

         ipython plt.py 
         ipython plt.py -pylab    ## huh without -pylab window does not pop up

   Plots all scans found in the .npz folder 


"""
import os
import numpy as np
import matplotlib
matplotlib.use('GTkAgg')               # Agg for non-interactive/web server usage
import matplotlib.pyplot as pyplot

fig = pyplot.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

npzs = lambda dir:filter(lambda _:_.endswith(".npz") , os.listdir(dir))

#styles = ("ro","go","bo", "r^", "g^", "b^", )

dir = ".npz"
for npz in npzs(dir):

    d = np.load(os.path.join(dir,npz)) 
    print "\n ========= %s =========== " % npz

    if not "meta" in d:continue
    if not "scan" in d:continue

    _meta = d["meta"]
    meta = dict(zip(_meta.dtype.names, _meta[0]))
    sc = d["scan"]

    print " meta : %s " % repr(meta)
    print "   sc : %s " % repr(sc)

    ax1.plot( sc['limit'], sc['ctime']  , meta.get("symbol","-")  ) 
    ax2.plot( sc['limit'], sc['rss_']   , meta.get("symbol","-")  ) 
    pass


fig.show()



