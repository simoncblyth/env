#!/usr/bin/env python
"""
Alternative Simple Backup using scp rather than rsync
=======================================================

This relies on the scm-backup machinery running on the source node
to create the source tarballs and dna sidecar files. The reason for
existance is that the scm rsyncing to SDU has been halted for
more than a month due to a bad target disk.  Requests to fix have 
not been successful.

I formerly rsynced to NTU, stopping when that got too time consuming
with growing tarball sizes.

Used a few changes to make offbox transfer more efficient:

#. use simple scp rather than rsync
#. restrict to transferring todays dybsvn tarballs only (exclude dybaux)

Possible further logic changes for efficiency:

#. trust the transfered digest sidecars 

TODO: 

#. notifications concerning completion

   * experience suggests best to do that in a separate cron job on target node

#. prune empty old directories when purging 


Deployment as cron task on WW
-------------------------------

The separate scm backup managed by Qiumei (needs to run as root) 
typically completes around 13:00 so time should be moved to later 
incase of slow backups 

::

	SHELL=/bin/bash
	HOME=/home/blyth
	ENV_HOME=/home/blyth/env
	CRONLOG_DIR=/home/blyth/cronlog
	NODE_TAG_OVERRIDE=WW
	#
	50 13 * * * ( . $ENV_HOME/env.bash ; env- ; python- source ; ssh-- ; cd $ENV_HOME/scm ; ./altbackup.py ) > $CRONLOG_DIR/altbackup.log 2>&1


"""
import logging, os, sys, stat, pprint
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

def find_todays_files( source, wanted=[], ext='.tar.gz'):
    """
    :param source: local directory 
    :param wanted: list of tarball names prefixes to return
    :return: list of tarball paths relative to source 
    """
    today = datetime.now().strftime("%Y/%m/%d")
    log.info("looking for %s source tarballs beneath %s from %s " % (repr(wanted), source, today) )
    tgzs = []
    for path, name in findpath(source, lambda p:p[-len(ext):] == ext and today in p):
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

def do(cmd, verbose=False, stderr=True):
    if not stderr:
        cmd = cmd + " 2>/dev/null"
    if verbose:
        print cmd 
    log.debug("do %s " % cmd )
    p = os.popen(cmd,'r')
    ret = p.read().strip()
    rc = p.close()
    log.debug("rc:%s len(ret):%s\n[%s]" % ( rc, len(ret), ret ))
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
 
def interpret_as_dna( ret ):
    assert ret[0] == '{' and ret[-1] == '}', "DNA has mutated %s     " % ret
    dna = eval(ret)
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

def remote_dna( rpath, node, cheat=False, sidecar_ext='.dna' ):
    """
    :param rpath: remote path
    :param node: remote node
    :param cheat: when True simply cat the sidecar otherwise calculate digest and size
    :param sidecar_ext: typically .dna
    """
    log.debug("remote_dna of %(rpath)s on node %(node)s cheating %(cheat)s " % locals() )
    fmt = "%s"

    if cheat:
        cmds = ["ssh %(node)s cat %(rpath)s%(sidecar_ext)s "]
    else: 
        cmds = [ 
            "ssh %(node)s stat --format %(fmt)s %(rpath)s ",
            "ssh %(node)s md5sum %(rpath)s ",
           ]

    rdna = {}
    for cmd in cmds:
        rc, ret = do(cmd % locals(),verbose=False,stderr=False)
        if cheat:
            rdna = interpret_as_dna(ret)
        else:
            if 'stat --format' in cmd:
                rdna['size'] = interpret_as_int(ret)
            if 'md5sum' in cmd:
                rdna['dig'] = interpret_as_md5sum(ret) 
    return rdna if len(rdna) == 2 else None


def transfer( spath, tpath , targetnode="C" , sidecar_ext='' ):
    log.info("transfer %(spath)s %(tpath)s %(targetnode)s %(sidecar_ext)s " % locals() )
    tdir = dirname(tpath) 
    cmds = [
              "ssh %(targetnode)s \"mkdir -p %(tdir)s \" ", 
              "time scp %(spath)s%(sidecar_ext)s %(targetnode)s:%(tpath)s%(sidecar_ext)s ", 
           ]
    for cmd in cmds:
        rc, ret = do(cmd % locals(),verbose=True)



def alt_transfer( source, target, cfg ):
    """
    :param source: base directory holding tarballs
    :param target: base directory where tarballs are to be scp copied
    :param cfg: 
    """
    for relpath in find_todays_files(source, wanted=cfg.wanted.split(), ext=cfg.ext ):

        spath = join(source, relpath)
        xdna  = sidecar_dna(spath)
        sdna  = local_dna(spath)
        assert xdna == sdna , ("mismatch between sidecar and recomputed dna %s %s " % ( xdna, sdna ))

        tpath = join(target, relpath)
        tdna  = remote_dna(tpath, cfg.targetnode, cheat=True )

        if tdna == sdna:
            log.info("target already transferred with dna matched, nothing to do %s " % tdna )
        else:
            log.info("target not transferred OR dna mis-matched, proceed to transfer  " )
            transfer( spath, tpath , targetnode=cfg.targetnode )
            rdna  = remote_dna(tpath, cfg.targetnode, cheat=False )

            # only transfer the .dna sidecar when match is found
            # suspect this is may be over called too many times  
            if sdna == rdna:
                log.info("succeeded to transfer tarball %s => %s , now transfer dna sidecar" % ( spath, tpath ) ) 
                transfer( spath, tpath , targetnode=cfg.targetnode , sidecar_ext='.dna')
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

def rmfile_( subfold, path , targetnode, ext , echo=False ):
    assert path[-len(ext):] == ext and path[:len(subfold)] == subfold, "path sanity check fails for %s " % path
    cmd = "echo rm -f %(path)s" if echo else "rm -f %(path)s"     
    if targetnode == 'LOCAL':
        assert cmd[0:4] == "echo", "local testing must just echo" 
    rmd = rmd_(cmd, targetnode=targetnode)
    rc, ret = do( rmd % locals(), verbose=True, stderr=True )


def alt_purge_cat( catfold, cfg ):
    """
    :param catfold:  category folder in which to look for tarballs eg tracs svn repos folders
    :param cfg:

     NB the file path of tarballs is assumed to be of the form that 
     sorting the paths puts them into date order

    """
    if catfold[-6:] == 'LOCKED':
         log.info("skipping %s " % catfold )
         return

    subfolds = subfolds_( catfold, targetnode=cfg.targetnode )
    nsubfold = len(subfolds)
    ext = cfg.ext

    log.info("catfold %(catfold)s has %(nsubfold)s subfolders " % locals())
    for subfold in subfolds:
        paths = findfiles_( subfold, cfg.targetnode, cfg.ext )  
        paths = sorted(paths, reverse=True)  # NB reverse date order, most recent first 
        npath = len(paths)
        log.info("    subfold %(subfold)s has %(npath)s paths with ext %(ext)s " % locals())
        for i, path in enumerate(paths): 
            mrk = "D" if i+1 > cfg.keep else ""
            log.info("        %-5s %-3s %s " % (i+1, mrk, path))  
            if mrk == "D":
                rmfile_( subfold, path, cfg.targetnode, cfg.ext, cfg.echo )

def alt_purge( target, cfg ):
    """
    :param target:
    :param cfg:

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
    catfolds = subfolds_( target, targetnode=cfg.targetnode )
    ncatfold = len(catfolds)
    log.info("altpurge for target %(target)s looking into %(ncatfold)s catfolds %(catfolds)s " % locals() )
    for catfold in catfolds:
        alt_purge_cat( catfold, cfg  )  



def alt_check( dir, cfg ):
    """
    """
    wanted = cfg.wanted.split()
    log.info("alt_check %s %s " % ( dir, wanted ) )
    expect = dict(svnsetup=1,dybsvn=2,dybaux=2)

    for want in wanted:
        relpaths = find_todays_files(dir, wanted=[want],ext=cfg.ext )
        for relpath in relpaths:
            path = join(dir, relpath)
            xdna  = sidecar_dna(path)
            sdna  = local_dna(path)
            assert xdna == sdna , ("mismatch between sidecar and recomputed dna %s %s " % ( xdna, sdna ))
        npaths = len(relpaths)
        assert npaths == expect[want], "expecting %s paths for %s BUT got %s  " % (expect[want], want, relpaths ) 


def parse_args_(doc):
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=None )
    op.add_option("-f", "--logformat", default="%(asctime)s %(name)s %(levelname)-8s %(message)s" )
    op.add_option("-l", "--loglevel", default="INFO" )
    op.add_option("-s", "--source",  default = "/home/scm/backup/dayabay", help="directory on the source node" )
    op.add_option("-t", "--target",  default = "/data/var/scm/alt.backup/dayabay", help="directory on the target node"  )
    op.add_option("-x", "--ext",     default = ".tar.gz", help="file type being managed"  )
    op.add_option("-n", "--targetnode",  default = "C" )
    op.add_option("-k", "--keep",  default = 3 )
    op.add_option("-e", "--echo", action="store_true", default = False )
    op.add_option("-w", "--wanted",  default = "dybsvn svnsetup" )  #  currently excludes dybaux
    opts, args = op.parse_args()
    level=getattr(logging,opts.loglevel.upper()) 

    if opts.logpath:
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
    else:
        logging.basicConfig(format=opts.logformat,level=level)
    return opts, args


if __name__ == '__main__':
    
    cfg, args = parse_args_(__doc__)

    source = cfg.source
    target = cfg.target
    cfg.source = None    # convenient for getting the values but not subsequently 
    cfg.target = None
    allowed = "dump transfer purge_target check_target check_source".split()
            
    if len(args) == 0: 
        print "expecting arguments such as %s " % allowed
        sys.exit(0)

    for arg in args:
        assert arg in allowed, "arg %s is not allowed %s " % ( arg, allowed )
        if arg == 'transfer':
            alt_transfer( source, target, cfg )   
        elif arg == 'purge_target':
            alt_purge(  target, cfg )   
        elif arg == 'check_source':
            alt_check( source, cfg )
        elif arg == 'check_target':
            node = os.environ['NODE_TAG']
            assert node == cfg.targetnode, "%s is running on node %s : should be run on targetnode %s  " % ( arg , node, cfg.targetnode )
            alt_check( target, cfg )
        elif arg == 'dump':
            log.info(pprint.pformat(cfg))
            log.info("source     : %(source)s " % locals())
            log.info("target     : %(target)s " % locals())
        else:
            pass



