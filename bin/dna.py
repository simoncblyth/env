#!/usr/bin/env python
"""
This duplicates code from env/scm/altbackup.py 

TODO: avoid the duplication
"""
import logging, os, sys, md5, stat
log = logging.getLogger(__name__)


def interpret_as_dna( ret ):
    if len(ret) > 0 and ret[0] == '{' and ret[-1] == '}':
        dna = eval(ret)
    else:
        dna = None
    log.debug("interpret_as_dna %s => %s " % ( ret, dna ))
    return dna

def sidecar_dna(path):
    """
    Reads the sidecar
    """
    sdna = open(path+'.dna','r').read().strip()
    return interpret_as_dna(sdna)

def local_dna( path ):
    log.debug("local_dna  determine size and digest of %s " % path )
    hash = md5.md5()
    size = 64*128   # 8192
    st = os.stat(path)
    sz = st[stat.ST_SIZE]
    f = open(path,'rb') 
    for chunk in iter(lambda: f.read(size), ''): 
        hash.update(chunk)
    f.close()
    dig = hash.hexdigest()
    dna = dict(dig=dig,size=int(sz))
    return dna 

def check_dna( path ):
    local = local_dna(path)
    sidecar = sidecar_dna(path)
    log.debug("local   %s\nsidecar %s" % (repr(local),repr(sidecar)) )
    assert local == sidecar 

def main():
    logging.basicConfig(level=logging.INFO)
    check_dna(sys.argv[1])

if __name__ == '__main__':
    main()




