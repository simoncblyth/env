#!/usr/bin/env python
"""
Alternative Simple Backup using scp rather than rsync
=======================================================

Suspect more digest checks than needed are being done, trusting the 
remote sidecar should reduce this a bit.

TODO: target purge/retain last 5 or whatever to prevent filling disk 

"""
import logging, os, subprocess, shlex, stat
log = logging.getLogger(__name__)
from os.path import join, getsize, dirname
from datetime import datetime
try: 
    from hashlib import md5
except ImportError: 
    from md5 import md5


def findpath(base, pathfilter=lambda _:True):
    for root, dirs, files in os.walk(base):
        for name in files:
            path = join(root,name)
            if pathfilter(path):
                yield path, name

def find_todays_tarballs( source, wanted=[]):
    """
    :param source: local directory 
    :param wanted: list of tarball names prefixes to return
    :return: list of tarball paths relative to source 
    """
    today = datetime.now().strftime("%Y/%m/%d")
    log.info("looking for %s source tarballs beneath %s from %s " % (repr(wanted), source, today) )
    tgzs = []
    for path, name in findpath(source, lambda p:p[-7:] == '.tar.gz' and today in p):
        if not os.path.exists(path+'.dna'):
            log.warn("SKIPPING AS no dna for path %s " % path )
            continue
        wnames = filter(lambda _:name.startswith(_), wanted )  # check if the name matches any of the wanted ones
        if len(wnames) == 0:
            log.debug("skip name %s " % name )
            continue
        spath = path[len(source)+1:]
        tgzs.append(spath)
    log.info("found %s matching tarballs" % len(tgzs) )
    return tgzs 

def do0(cmd):
    elem = shlex.split(cmd)
    print elem
    p= subprocess.Popen(elem, shell=True, stdout=subprocess.PIPE )
    ret = p.stdout.read()
    rc = p.stdout.close()
    return rc, ret  

def do(cmd, verbose=False):
    p = os.popen(cmd,'r')
    ret = p.read()
    rc = p.close()
    if verbose:
        print "rc:%s ret:%s " % ( rc, ret )
    return rc, ret


def interpret_as_int(ret):
    ival=None
    if ret:
        try:
            ival = int(ret)
        except ValueError:
            pass
    return ival 

def interpret_as_md5sum(ret):
    dval = None
    elem = ret.strip().split()
    if len(elem) == 2:
        dval = elem[0]
    else:
        log.debug("failed to inerpret %s as md5sum " % ret )
    return dval 
         

def sidecar_dna(path):
    """
    Reads the sidecar
    """
    sdna = open(path+'.dna','r').read().strip()  
    assert sdna[0] == '{' and sdna[-1] == '}', "DNA has mutated %s     " % sdna
    dna = eval(sdna)
    return dna

def source_dna( path ):
    hash = md5()
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

def target_dna( tpath, targetnode="C" ):
    fmt = "%s"
    chks = [ 
            "ssh %(targetnode)s stat --format %(fmt)s %(tpath)s ",
            "ssh %(targetnode)s md5sum %(tpath)s ",
           ]
    tdna = {}
    for cmd in chks:
        cmd = cmd % locals()
        print cmd
        rc, ret = do(cmd,verbose=True)
        if 'stat --format' in cmd:
            tdna['size'] = interpret_as_int(ret)
        if 'md5sum' in cmd:
            tdna['dig'] = interpret_as_md5sum(ret) 
    return tdna if len(tdna) == 2 else None


def transfer( spath, tpath , targetnode="C" , ext='' ):
    tdir = dirname(tpath) 
    cmds = [
              "ssh %(targetnode)s \"mkdir -p %(tdir)s \" ", 
              "time scp %(spath)s%(ext)s %(targetnode)s:%(tpath)s%(ext)s ", 
           ]
    for cmd in cmds:
        cmd = cmd % locals()
        print cmd
        rc, ret = do(cmd,verbose=True)


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)

    #wanted = "dybsvn dybaux svnsetup"
    #wanted = "svnsetup"
    wanted = "dybsvn"
    source = "/home/scm/backup/dayabay"
    target = "/data/var/scm/alt.backup/dayabay"
    targetnode = "C"        

    for relpath in find_todays_tarballs(source, wanted=wanted.split()):

        # source 
        spath = join(source, relpath)
        xdna  = sidecar_dna(spath)
        sdna  = source_dna(spath)
        assert xdna == sdna , ("mismatch between sidecar and recomputed dna %s %s " % ( xdna, sdna ))

        # target 
        tpath = join(target, relpath)
        tdna  = target_dna(tpath, targetnode=targetnode )

        if tdna and tdna == sdna:
            log.info("target already transferred with dna matched, nothing to do %s " % tdna )
            continue

        transfer( spath, tpath , targetnode=targetnode )
        rdna  = target_dna(tpath, targetnode=targetnode )

        # only transfer the .dna sidecar when match is found
        if sdna == rdna:
            log.info("succeeded to transfer tarball %s => %s , now transfer dna sidecar" % ( spath, tpath ) ) 
            transfer( spath, tpath , targetnode=targetnode , ext='.dna')
        else:
            log.warn("dna mismatch %s %s " % (sdna, rdna))
        

