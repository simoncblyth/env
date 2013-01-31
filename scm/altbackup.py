#!/usr/bin/env python
"""
Alt backup via scp to C
"""
import logging, os, subprocess, shlex
log = logging.getLogger(__name__)
from os.path import join, getsize, dirname
from datetime import datetime

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
    log.info("looking for source tarballs beneath %s " % source )
    today = datetime.now().strftime("%Y/%m/%d")
    tgzs = []
    for path, name in findpath(source, lambda p:p[-7:] == '.tar.gz' and today in p):
        if not os.path.exists(path+'.dna'):
            log.warn("SKIPPING AS no dna for path %s " % path )
            continue
        wnames = filter(lambda _:name.startswith(_), wanted )  # check if the name matches any of the wanted ones
        if len(wnames) == 0:
            log.info("skip name %s " % name )
            continue
        pass
        spath = path[len(source)+1:]
        tgzs.append(spath)
    return tgzs 


def do0(cmd):
    elem = shlex.split(cmd)
    print elem
    p= subprocess.Popen(elem, shell=True, stdout=subprocess.PIPE )
    ret = p.stdout.read()
    rc = p.stdout.close()
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
        log.warn("failed to inerpret %s as md5sum " % ret )
    return dval 
         


def do(cmd, verbose=False):
    p = os.popen(cmd,'r')
    ret = p.read()
    rc = p.close()
    if verbose:
        print "rc:%s ret:%s " % ( rc, ret )
    return rc, ret

def source_dna(path, checksize=False, checkdigest=False):
    """
    Reads the sidecar, hmm maybe should do the sum 
    """
    sdna = open(path+'.dna','r').read()  
    dna = eval(sdna)
    if checksize:
        ssize = getsize(path)
        dsize = dna['size']
        assert ssize == dsize , ("size mismatch between a tarball DNA record and file system", ssize, dsize, path )
        log.info(" %-50s %s %s " % (path, ssize, dsize )) 
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
        spath = join(source, relpath)
        tpath = join(target, relpath)

        sdna  = source_dna(spath, checksize=True, checkdigest=False )
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
        

