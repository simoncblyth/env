#!/usr/bin/env python
"""
Refractive Index
=================

* https://en.wikipedia.org/wiki/Dispersive_prism
* https://en.wikipedia.org/wiki/Flint_glass
* http://refractiveindex.info/tmp/glass/schott/F2.csv
* https://en.wikipedia.org/wiki/Abbe_number

Crown glasses such as BK7 have a relatively small dispersion, while flint
glasses have a much stronger dispersion (for visible light) and hence are more
suitable for use in dispersive prisms.

::

    import env.physics.refractiveindex.refractiveindex as ri
    a = ri.refractiveindex("tmp/glass/schott/F2.csv")

    plt.plot(a[:,0],a[:,1])
    plt.show()

"""
import numpy as np
import os, logging, urllib2

log = logging.getLogger(__name__)

def retrieve(relp, urlbase, cache):
    url = os.path.join(urlbase, relp) 
    path = os.path.join(cache, relp)

    if os.path.exists(path):
        log.info("read from path %s" % path) 
        c = file(path).read()
    else: 
        log.info("download from url %s" % url) 
        p = urllib2.urlopen(url)
        c = p.read()

        dirp = os.path.dirname(path)
        if not os.path.exists(dirp):
            os.makedirs(dirp)

        with open(path, "wb") as fp:
            fp.write(c)
        pass
    return c 


def parse(csv):
    """
    Remove annoying second table, newline chars
    """ 
    firsttab = csv.split("wl,k")[0]
    lines = filter(None,firsttab.split("\r\n"))
    txt = ",".join(lines[1:])
    a = np.fromstring(txt, sep=",", dtype=np.float32).reshape(-1,2)
    a[:,0] *= 1000.  # to nm 
    return a 

def refractiveindex(rel, base=None, cache=None):
    """
    """

    if base is None:
        base = "http://refractiveindex.info"
    if cache is None:
        #cache = os.path.expandvars("$LOCAL_BASE/env/physics/refractiveindex")
        cache = os.path.expanduser("~/opticksdata/refractiveindex")

    nrel = rel.replace(".csv",".npy")
    npath = os.path.join(cache, nrel)
    if os.path.exists(npath):
        log.info("loading from %s " % npath )
        a = np.load(npath)
    else:
        csv = retrieve(rel, base, cache)
        log.info("parsing from %s " % rel )
        a = parse(csv)
        np.save(npath, a)    
    pass
    return a 


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    a = refractiveindex("tmp/glass/schott/F2.csv")
    b = refractiveindex("tmp/main/H2O/Hale.csv") 

    log.info("a:%s" % repr(a.shape))
    log.info("b:%s" % repr(b.shape))
    


