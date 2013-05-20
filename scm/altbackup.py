#!/usr/bin/env python
"""
Alternative Simple Backup using scp rather than rsync
=======================================================

Multiple command arguments are accepted togther with configuration options::

   altbackup.py --help
   altbackup.py check_source
   altbackup.py dump check_source transfer purge_target

On target node only::

   altbackup.py check_target              # digest recomputation and comparison against sidecar dna
   altbackup.py extract_tracdb            # todays tarball if already copied over
   altbackup.py extract_tracdb --day -1   # yesterdays tarball
   altbackup.py extract_tracdb --day 2013/04/13    
         # specific tarball, only a few days tarballs are retained

   altbackup.py examine_tracdb --day -1   # yesterdays tarball


Available commands:

*transfer*
      scp matched files from source to target 

*purge_target*
      deletes remote files with the subfold on the targetnode 
      retaining only the last `cfg.keep` files within each <catfold>/<subfold>
      Also deletes empty remote directories within the subfold, 
      only one search for empties is made within each subfold
      so repeated invokation is typically needed to purge all empty directories. 

      **NB assumes lexically sorted file paths are in date order**  

*check_source*
      find matched files and 
      checks that the sidecar dna matches the locally recomputed dna


Commands which can be run on the configured targetnode only:

*check_target*
      checks that the sidecar dna matches the locally recomputed dna

*extract_tracdb*
      extract trac.db out of the last backup tarball on the targetnode

*examine_tracdb*
      examine trac.db out of the last backup tarball, printing last build times from each of the slaves


*dump*
      print configuration parameters

*tls* OR *ls*
      list target tarballs on targetnode (unenforced)

*sls* 
      list source tarballs on sourcenode (unenforced)

Timings and cron invokation
----------------------------

See commentry in the bash wrapper script `altbackup.sh` that invokes this python script 
and handles email notifications of non=zero return codes.


Issues
--------

truncated transfer due to connection timeout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    [blyth@cms01 05]$ pwd
    /data/var/scm/alt.backup/dayabay/tracs/dybsvn/2013/05

    [blyth@cms01 05]$ find . -type f  -name '*.tar.gz' -exec ls -l {} \;
    -rw-r--r--  1 blyth blyth 1529629254 May  8 14:06 ./08/104701/dybsvn.tar.gz
    -rw-r--r--  1 blyth blyth 1531229474 May  9 14:01 ./09/104702/dybsvn.tar.gz
    -rw-r--r--  1 blyth blyth 1147682816 May 10 14:08 ./10/104701/dybsvn.tar.gz




"""

notes=r"""
Notes
--------

This relies on the scm-backup machinery running on the source node
to create the source tarballs and dna sidecar files. 
This exists as the scm rsyncing to SDU has been halted for
more than a month due to a bad target disk.  
Requests to Shandong colleages to fix or find and alterative
target have not been successful.

I formerly rsynced with the scm-backup to NTU, 
Only stopping when the time to complete was approaching 24hrs 
with growing tarball sizes.

Used a few changes to make offbox transfer more efficient:

#. use simple scp rather than rsync
#. restrict to transferring todays dybsvn tarballs only (exclude dybaux)
#. possibly some throttling might be encountered

TODO 
-----

#. time and size logging to sqlite to allow plotting, in same way as scm-backup does 
#. prune empty old directories when purging 

#. add `purge` command to remove old extracted trac.db::

	[blyth@cms01 dybsvn]$ pwd
	/data/env/tmp/tracs/dybsvn
	[blyth@cms01 dybsvn]$ find .  -name 'trac.db' -exec du -h {} \;
	6.7G    ./2013/04/25/104702/dybsvn/db/trac.db
	6.7G    ./2013/05/09/104702/dybsvn/db/trac.db
	6.7G    ./2013/05/16/104702/dybsvn/db/trac.db
	[blyth@cms01 dybsvn]$ rm -f  ./2013/04/25/104702/dybsvn/db/trac.db
	[blyth@cms01 dybsvn]$ rm -f  ./2013/05/09/104702/dybsvn/db/trac.db


Deployment as cron tasks on source and target
-----------------------------------------------

This is done via script altbackup.sh see usage notes within that.

"""
import logging, os, sys, stat, pprint
import tarfile # new in 2.3
assert tuple(sys.version_info)[0:2] in [(2,4),(2,5),(2,6),(2,7)] , "unexpected python version %s \n%s" % ( repr(sys.version_info) , __doc__ )
# (2,4) might work for sphinx doc building only 
log = logging.getLogger(__name__)
from os.path import join, getsize, dirname
from datetime import datetime, timedelta
try: 
    from hashlib import md5
except ImportError: 
    from md5 import md5


seconds = {}

def timing(func):
    def wrapper(*arg,**kw):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        t1 = time.time()
        res = func(*arg,**kw)
        t2 = time.time()
        global seconds
        seconds[func.func_name] = (t2-t1)
        return res 
    return wrapper


def findpath(base, pathfilter=lambda _:True):
    for root, dirs, files in os.walk(base):
        for name in files:
            path = join(root,name)
            if pathfilter(path):
                yield path, name

def find_day_files( source, wanted=[], ext='.tar.gz', day=None):
    """
    :param source: local directory 
    :param wanted: list of tarball names prefixes to return
    :param ext: file extension, defaults to .tar.gz
    :param day: date string in format 2013/04/13, defaults to todays date
    :return: list of tarball paths relative to source 
    """
    assert day
    log.info("looking for %s source tarballs beneath %s from %s " % (repr(wanted), source, day) )
    tgzs = []
    for path, name in findpath(source, lambda p:p[-len(ext):] == ext and day in p):
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

def ls_(dir, ext=".tar.gz"):
    cmd = "find %(dir)s -name '*%(ext)s' -exec ls -lh {} \;" % locals()
    log.info(cmd)
    log.info("\n"+os.popen(cmd).read())

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
    sret = ret.strip()
    if len(sret) == 0:
        elem = []
    else:
        elem = sret.split(delim)
    return elem
 
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
    if rdna and len(rdna) == 2:
        return rdna
    return None


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
    for relpath in find_day_files(source, wanted=cfg.wanted.split(), ext=cfg.ext, day=cfg.day ):

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
    if targetnode != "LOCAL":
        rmd = "ssh %(targetnode)s \"" + cmd + "\"" 
    else:
        rmd = cmd
    return rmd

def find_( basefold, targetnode, condition , verbose=False):
    cmd = "find %(basefold)s " + condition 
    rmd = rmd_(cmd, targetnode=targetnode)
    rc, ret = do( rmd % locals(), verbose=verbose, stderr=True )
    paths = interpret_as_linelist(ret, delim="\n")
    return paths

def findfiles_( basefold, targetnode, ext ):
    return find_( basefold, targetnode, " -name '*" + ext + "' "  )    
def findempty_(basefold, targetnode , verbose=False):
    return find_( basefold, targetnode, "  -type d -empty " , verbose=verbose )

def subfolds_( basefold, targetnode ):
    cmd = "ls -1d %(basefold)s/*" 
    rmd = rmd_(cmd, targetnode=targetnode)
    rc, ret = do(rmd % locals(),verbose=False, stderr=False)   
    subfolds = interpret_as_linelist(ret, delim="\n")
    return subfolds

def rmfile_( subfold, path , targetnode, ext , echo=False ):
    assert path[-len(ext):] == ext and path[:len(subfold)] == subfold, "path sanity check fails for %s " % path
    cmd = echo_("rm -f %(path)s",echo)     
    if targetnode == 'LOCAL':
        assert cmd[0:4] == "echo", "local testing must just echo" 
    rmd = rmd_(cmd, targetnode=targetnode)
    rc, ret = do( rmd % locals(), verbose=True, stderr=True )

def echo_(cmd, echo):
    if echo: 
        ret = "echo " + cmd 
    else:
        ret = cmd
    return ret

def rmdir_( subfold, path, targetnode, echo=False ):
    assert path[:len(subfold)] == subfold, "path sanity check fails for %s " % path
    cmd = echo_("rmdir %(path)s",echo)     
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
        edirs = findempty_( subfold, cfg.targetnode , verbose=True )  
        nedir = len(edirs)
        log.info("    subfold %(subfold)s has %(npath)s paths with ext %(ext)s empty dirs %(nedir)s" % locals())
        for i, path in enumerate(paths): 
            if i+1 > cfg.keep:
                mrk = "D"
            else:
                mrk = ""
            log.info("        %-5s %-3s %s " % (i+1, mrk, path))  
            if mrk == "D":
                rmfile_( subfold, path, cfg.targetnode, cfg.ext, cfg.echo )
                rmfile_( subfold, path + '.dna', cfg.targetnode, cfg.ext + '.dna', cfg.echo )

        for edir in sorted(edirs):
            log.debug("empty folder [%s] " % edir )
            rmdir_( subfold, edir, cfg.targetnode, cfg.echo ) 



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
    log.info("alt_check %s %s : checking sidecar dna matches locally recomputed   " % ( dir, wanted ) )
    expect = dict(svnsetup=1,dybsvn=2,dybaux=2)

    for want in wanted:
        relpaths = find_day_files(dir, wanted=[want],ext=cfg.ext, day=cfg.day  )
        for relpath in relpaths:
            path = join(dir, relpath)
            xdna  = sidecar_dna(path)
            sdna  = local_dna(path)
            assert xdna == sdna , ("mismatch between sidecar and recomputed dna %s %s " % ( xdna, sdna ))
        npaths = len(relpaths)
        assert npaths == expect[want], "expecting %s paths for %s BUT got %s  " % (expect[want], want, relpaths ) 


def alt_examine_tracdb( dir, cfg ):
    for trac in cfg.wanted.split():
        for relpath in find_day_files(dir, wanted=[trac],ext=cfg.ext, day=cfg.day  ):
            fold = os.path.join( cfg.base, os.path.dirname(relpath) )
            dbpath = os.path.join(fold,"%s/db/trac.db" % trac)
            if not os.path.exists(dbpath):
                log.warn("skip non existing %s " % dbpath )
            else:
                log.info("dbpath %s " % dbpath)
                query( dbpath )

def query(dbpath):
    """
    Use the trac.db extracted from the backup tarball to determine the laststart times of the last build
    from each of the slaves, query restricts to last 30 days to skip retired slaves::


	slave                 id          rev         lastrevtime           laststarted         
	--------------------  ----------  ----------  --------------------  --------------------
	lxslc507.ihep.ac.cn   20630       20453       2013-05-02 11:43:14   2013-05-02 12:04:01 
	farm4.dyb.local       20629       20453       2013-05-02 11:43:14   2013-05-02 12:07:55 
	lxslc504.ihep.ac.cn   20626       20453       2013-05-02 11:43:14   2013-05-02 13:03:01 
	belle7.nuu.edu.tw     20652       20479       2013-05-08 00:23:01   2013-05-08 03:40:09 
	daya0001.rcf.bnl.gov  20654       20479       2013-05-08 00:23:01   2013-05-08 08:43:41 
	daya0004.rcf.bnl.gov  20658       20479       2013-05-08 00:23:01   2013-05-08 09:20:28 
	pdyb-02               20661       20479       2013-05-08 00:23:01   2013-05-08 09:48:35 
	farm2.dyb.local       20655       20479       2013-05-08 00:23:01   2013-05-08 11:22:42 
	pdyb-03               20657       20479       2013-05-08 00:23:01   2013-05-09 00:13:52 

    """ 
    dt_ = lambda field,label:"datetime(%(field)s,'unixepoch') as %(label)s" % locals()
    cols = ",".join(["slave","max(id) id", "max(rev) rev", "max(rev_time) as _lastrevtime", dt_("max(rev_time)", "lastrevtime"), "max(started) as _laststart", dt_("max(started)","laststart") ])
    keys = ("slave","id","rev","_lastrevtime","lastrevtime","_laststart","laststart",)
    modifier = "-30 days"
    sql = "select %(cols)s from bitten_build where datetime(rev_time,'unixepoch') > datetime('now','%(modifier)s') group by slave having slave != '' order by _laststart " % locals()
    
    #print "\n"+os.popen("echo \"%(sql)s ; \" | sqlite3 -header -column  %(dbpath)s " % locals()).read()  # with the sqlite3 binary 
    import sqlite3 
    conn = sqlite3.connect(dbpath)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    fmt = " %(slave)-20s  %(id)-7s %(rev)-7s     %(lastrevtime)-20s    %(laststart)-20s  " 
    print fmt % dict((k,k) for k in keys)
    for row in cursor.execute(sql):
        d = dict((k,row[k]) for k in keys)
        print fmt % d

def alt_extract_tracdb( dir, cfg ):
    """
    """
    for trac in cfg.wanted.split():
        relpaths = find_day_files(dir, wanted=[trac],ext=cfg.ext, day=cfg.day  )
        for relpath in relpaths:
            path = join(dir, relpath)
            name = os.path.basename(path)
            if name == "%s%s" % (trac,cfg.ext):
                fold = os.path.join( cfg.base, os.path.dirname(relpath) )
                tracdb = "%s/db/trac.db" % trac
                dbpath = os.path.join(fold,tracdb)
                if os.path.exists(dbpath):
                    log.info("already extracted %s nothing to do " % dbpath )
                else:
                    log.info("extract_tracdb path %s relpath %s " % (path,relpath) )
                    tgz = tarfile.open(path,"r")
                    if not os.path.exists(fold):
                        log.info("create output folder for extraction %s " % fold )
                        os.makedirs(fold)
                    try:
                        tin = tgz.getmember(tracdb)
                    except KeyError:
                        tin = None
                        log.info("failed to find %s in the tarball " % tracdb )
                    if not tin:
                        log.warn("skip extraction ")
                    else: 
                        log.info("extracting %s into %s this will take a while " % (tracdb, fold ))
                        tgz.extract(tracdb, path=fold )
                        assert os.path.exists(dbpath)
                        log.info("completed extraction to %s " % dbpath )
                pass
                if os.path.exists(dbpath):
                    log.info( os.popen("du -h %(dbpath)s" % locals()).read() )
            else:
                log.info("exclude tarball %s " % path )






def interpret_day( s ):
    """
    :param s: None for today, "-1" "-2" "-3" for days past, OR "2013/04/13" for specific day
    :return: date string in standard format
    """
    tfmt = "%Y/%m/%d" 
    now = datetime.now()
    if not s:
        t = now
    elif s[0] == "-":
        t = now + timedelta(days=int(s))    
    else:
        t = datetime.strptime(s,tfmt)
    ss = t.strftime(tfmt)
    log.info("interpreted day string %s into %s " % (s,ss))
    return ss


def parse_args_(doc):
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-o", "--logpath", default=None )
    op.add_option("-f", "--logformat", default="%(asctime)s %(name)s %(levelname)-8s %(message)s" )
    op.add_option("-l", "--loglevel", default="INFO" )
    op.add_option("-s", "--source",  default = "/home/scm/backup/dayabay", help="directory on the source node" )
    op.add_option("-t", "--target",  default = "/data/var/scm/alt.backup/dayabay", help="directory on the target node"  )
    op.add_option("-d", "--day",  default = None, help="date string in format '2013/04/13' or default of None corresponding to today or -1 for yesterday, -2 for day before yesterday"  )
    op.add_option("-x", "--ext",     default = ".tar.gz", help="file type being managed"  )
    op.add_option("-n", "--targetnode",  default = "C" )
    op.add_option("-k", "--keep",  default = 3 )
    op.add_option("-b", "--base",  default = "/data/env/tmp", help="base directory under which extracted trac.db are placed")
    op.add_option("-e", "--echo", action="store_true", default = False )
    op.add_option("-w", "--wanted",  default = "dybsvn svnsetup" )  #  currently excludes dybaux
    opts, args = op.parse_args()
    level=getattr(logging,opts.loglevel.upper()) 
    if opts.logpath:
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
    else:
        logging.basicConfig(format=opts.logformat,level=level)
    pass
    log.info(" ".join(sys.argv))
    opts.day = interpret_day(opts.day) 
    return opts, args


def main():    
    cfg, args = parse_args_(__doc__)

    source = cfg.source
    target = cfg.target
    cfg.source = None    # convenient for getting the values but not subsequently 
    cfg.target = None
    allowed = "dump transfer purge_target check_target check_source extract_tracdb examine_tracdb tls sls ls".split()
            
    if len(args) == 0: 
        print "expecting arguments such as %s " % allowed
        sys.exit(0)

    for arg in args:
        assert arg in allowed, "arg %s is not allowed %s " % ( arg, allowed )
        log.info("================================ %s " % arg )
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
        elif arg == 'extract_tracdb':
            node = os.environ['NODE_TAG']
            assert node == cfg.targetnode, "%s is running on node %s : should be run on targetnode %s  " % ( arg , node, cfg.targetnode )
            alt_extract_tracdb( target, cfg )
        elif arg == 'examine_tracdb':
            node = os.environ['NODE_TAG']
            assert node == cfg.targetnode, "%s is running on node %s : should be run on targetnode %s  " % ( arg , node, cfg.targetnode )
            alt_examine_tracdb( target, cfg )
        elif arg == 'tls' or arg == 'ls':
            ls_(target) 
        elif arg == 'sls':
            ls_(source) 
        elif arg == 'dump':
            log.info(pprint.pformat(cfg))
            log.info("source     : %(source)s " % locals())
            log.info("target     : %(target)s " % locals())
        else:
            pass


if __name__ == '__main__':
    main()

