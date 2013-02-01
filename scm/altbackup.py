#!/usr/bin/env python
"""
Alternative Simple Backup using scp rather than rsync
=======================================================

Suspect more digest checks than needed are being done, trusting the 
remote sidecar should reduce this a bit.


TODO: prune empty old directories when purging 


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

def do_not_working(cmd):
    elem = shlex.split(cmd)
    print elem
    p= subprocess.Popen(elem, shell=True, stdout=subprocess.PIPE )
    ret = p.stdout.read()
    rc = p.stdout.close()
    return rc, ret  

def do(cmd, verbose=False, stderr=True):
    if not stderr:
        cmd = cmd + " 2>/dev/null"
    if verbose:
        print cmd 
    p = os.popen(cmd,'r')
    ret = p.read().strip()
    rc = p.close()
    if verbose:
        print "rc:%s len(ret):%s\n[%s]" % ( rc, len(ret), ret )
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
         
def interpret_as_linelist(ret, delim="\n"):
    elem = ret.strip().split(delim)
    return elem
 
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
        rc, ret = do(cmd % locals(),verbose=True)
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
        rc, ret = do(cmd % locals(),verbose=True)



def altbackup( source, target, targetnode, wanted ):
    """
    :param source: base directory holding tarballs
    :param target: base directory where tarballs are to be scp copied
    :param targetnode: ssh config name eg C
    :param wanted: list of tarball name prefixes
    """
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


def rmd_(cmd, targetnode):
    rmd = "ssh %(targetnode)s \"" + cmd + "\"" if targetnode != "LOCAL" else cmd
    return rmd

def findfiles_( basefold, targetnode, ext ):
    cmd = "find %(basefold)s -name '*" + ext + "' "    
    rmd = rmd_(cmd, targetnode=targetnode)
    rc, ret = do( rmd % locals(), verbose=False, stderr=True )
    tgzs = interpret_as_linelist(ret, delim="\n")
    return tgzs

def subfolds_( basefold, targetnode ):
    cmd = "ls -1d %(basefold)s/*" 
    rmd = rmd_(cmd, targetnode=targetnode)
    rc, ret = do(rmd % locals(),verbose=False, stderr=False)   
    subfolds = interpret_as_linelist(ret, delim="\n")
    return subfolds

def rmfile_( subfold, path , targetnode, ext ):
    assert path[-len(ext):] == ext and path[:len(subfold)] == subfold, "path sanity check fails for %s " % path
    #cmd = "echo rm -f %(path)s"     
    cmd = "rm -f %(path)s"     
    if targetnode == 'LOCAL':
        assert cmd[0:4] == "echo", "local testing must just echo" 
    rmd = rmd_(cmd, targetnode=targetnode)
    rc, ret = do( rmd % locals(), verbose=True, stderr=True )


def altpurge_cat( catfold, targetnode, ext='.tar.gz', keep=3 ):
    """
    :param catfold:  category folder in which to look for tarballs eg tracs svn repos folders
    :param targetnode:
    :param keep: number of tarballs to keep of each variety

     NB the file path of tarballs is assumed to be of the form that 
     sorting the paths puts them into date order

    """
    if catfold[-6:] == 'LOCKED':
         log.info("skipping %s " % catfold )
         return

    subfolds = subfolds_( catfold, targetnode=targetnode )
    nsubfold = len(subfolds)

    log.info("catfold %(catfold)s has %(nsubfold)s subfolders " % locals())
    for subfold in subfolds:
        paths = findfiles_( subfold, targetnode, ext )  
        paths = sorted(paths, reverse=True)  # NB reverse date order, most recent first 
        npath = len(paths)
        log.info("    subfold %(subfold)s has %(npath)s paths with ext %(ext)s " % locals())
        for i, path in enumerate(paths): 
            mrk = "D" if i+1 > keep else ""
            log.info("        %-5s %-3s %s " % (i+1, mrk, path))  
            if mrk == "D":
                rmfile_( subfold, path, targetnode, ext )

def altpurge( target, targetnode, ext='.tar.gz', keep=3 ):
    """
    :param target:
    :param targetnode:
    :param keep:

    NB this method together with altpurge_cat assumes a two level hierarchy to the
    directory structure used to keep the tarballs. With purging taking place amongst the 
    set of tarballs found within the subfold.

    ::
 
          target/<catfold>/<subfold>/....
                 tracs      dybsvn 
                 svn        dybaux
                 repos
                 folders

    Within each category folder eg "tracs", "svn", "repos", "folders" 
    there are multiple subfolders eg "dybsvn", "dybaux", "env" and
    within those are day folders containing the leaf tarballs

    """
    catfolds = subfolds_( target, targetnode=targetnode )
    ncatfold = len(catfolds)
    log.info("altpurge for target %(target)s looking into %(ncatfold)s catfolds %(catfolds)s on targetnode %(targetnode)s keep %(keep)s " % locals() )
    for catfold in catfolds:
        altpurge_cat( catfold, targetnode, ext=ext, keep=keep )  


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO)

    #wanted = "dybsvn dybaux svnsetup"
    wanted = "svnsetup"
    #wanted = "dybsvn"
    source = "/home/scm/backup/dayabay"
    target = "/data/var/scm/alt.backup/dayabay"
    targetnode = "C"        

    altbackup( source, target, targetnode, wanted )   
    #altpurge( target, targetnode=targetnode, keep=3 )   ## real remote action 

    #altpurge( source, targetnode="LOCAL", keep=3 )       ## local testing


