#!/usr/bin/env python
"""
Usage::

    python tools/parallelize.py -h
    python tools/parallelize.py        # check 
    python tools/parallelize.py | sh   # do

Source and target directories are specified in config ini files, for example::

    [html2wdocs]
    object = propagate generated html from the Sphinx build into the wdocs where is can be made mobile 
    argv0 = parallelize.py
    source = ~/workflow/_build/html
    target = ~/DELTA/wdocs

An example application is propagation of Sphinx html build products
from emphemeral Sphinx build directories (that are deleted by doing "make clean")
into a more permanent directory structures, allowing derived html 
docs to be synced to mobile devices, and accessed offline.

#. This is a simple migrator between directory heirarchies, with digest checking 
   of whether files have been changed. There is no regard for SVN status.

#. The script does not actually do anything, just suggests shell commands to run 
   to be checked by user before piping them to sh 

"""
from __future__ import with_statement
import os, sys, hashlib, logging, re
log = logging.getLogger(__name__)

def digest2_(path, block_size=8192):
    """
    :param path:
    :return: md5 hexdigest of the content of the path or None if non-existing path

    http://stackoverflow.com/questions/1131220/get-md5-hash-of-a-files-without-open-it-in-python

    Confirmed to give same hexdigest as commandline /sbin/md5::

        md5 /Users/blyth/workflow/notes/php/property/colliers-4q2011.pdf 
        MD5 (/Users/blyth/workflow/notes/php/property/colliers-4q2011.pdf) = 3a63b5232ff7cb6fa6a7c241050ceeed

    """
    md5 = hashlib.md5()
    with open(path,'rb') as f: 
        for chunk in iter(lambda: f.read(block_size),''): 
            md5.update(chunk)
    return md5.hexdigest()


def digest3_(path, block_size=8192):
    dig = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            pass
            dig.update(chunk)
        pass
    pass
    return dig.hexdigest()


def digest_(path, pyver=0):
    """ 
    :param path:
    :param pyver:  
    :return: md5 hexdigest of the content of the path or None if non-existing path

    http://stackoverflow.com/questions/1131220/get-md5-hash-of-a-files-without-open-it-in-python

    Confirmed to give same hexdigest as commandline /sbin/md5::
    """

    if not os.path.exists(path):return None
    if os.path.isdir(path):return None

    hexdig = None
    if pyver == 0 or pyver == 3:
        hexdig = digest3_(path)
    elif pyver == 2:
        hexdig = digest2_(path)
    else:
        assert 0, "invalid pyver %d" % pyver
    pass
    return hexdig




class DPath(object):
    """
    Holder object for:

    #. path
    #. content digest
    """
    def __init__(self, path):
        """
        :param path: 
        """
        self.path = path 
        self.digest = digest_(path)
        self.isdir = os.path.isdir(path)
    def __repr__(self):
        return "%s %s DIR:%s " % (self.path, self.digest, self.isdir )     


class Parallelize(list):
    """
    Copying into parallel heirarchy.
    """
    def __init__(self, src, tgt, exts=None , skipd=None ):    
        """
        :param src: source directory 
        :param tgt: target directory
        :param exts: default of None corresponds to all, whitespace delimited list eg ".pdf .png" 
        :param skip: space delimited list of dir names to skip in the dir_walk eg "_build _sources"

        """
        xx_ = lambda _:os.path.expandvars(os.path.expanduser(_))
        self.src = xx_(src)
        self.tgt = xx_(tgt)

        self.cmds = []
        self.skipd = skipd.split() if skipd else []
        self.exts = exts.split() if exts else None

        self._exts = {}
        self._collect_exts(self.src)
        self._collect_exts(self.tgt)
        self.dir_walk(self.src, self.copy_handle )

    def _collect_exts(self, rootdir):
        """
        Collect all filetypes found by recursive walk from rootdir into 
        dict-of-set structure keyed by rootdir
        """
        self._exts[rootdir] = set() 
        def collect_(path):
            root, ext = os.path.splitext(path)
            self._exts[rootdir].add(ext)
            pass
        self.dir_walk(rootdir, collect_)
        log.info("collected %s exts from %s : %s " % ( len(self._exts[rootdir]), rootdir, self._exts[rootdir] ))

    def dir_walk(self, rootdir, pathfn_ ):
        """
        :param rootdir:
        :param pathfn_: function applied to every absolute path meeting selection

        Simple os.walk the source tree handling files with extensions matching the 
        selected ones.
        """
        for dirpath, dirs, names in os.walk(rootdir):
            rdir = dirpath[len(rootdir)+1:]    
            for skp in self.skipd:         
                if skp in dirs:
                    dirs.remove(skp)  
            for name in names:
                root, ext = os.path.splitext(name)
                if self.exts and not ext in self.exts: 
                    continue
                spath = os.path.join(dirpath, name)
                pathfn_(spath)
            pass    
        pass

    def copy_handle(self, spath):
        """
        :param spath: full source path 

        #. Determine target path from source path by prefix swapping
        #. Compare content digest of source and target files, if they 
           differ or the target does not exist prepare commands 
           to copy from source to target creating any needed directories
           and append these to `self.cmds`

        """ 
        rpath = spath[len(self.src)+1:]
        sp = DPath(spath) 
        tpath = os.path.join(self.tgt,rpath)
        tp = DPath(tpath) 
        if sp.digest == tp.digest:
            log.debug("no update needed %r " % sp )         
        else:   
            log.debug("sp %s " % ( sp ))
            log.debug("tp %s " % ( tp ))
            cmd = "mkdir -p \"%s\" && cp \"%s\" \"%s\" " % ( os.path.dirname(tpath), spath, tpath )
            self.cmds.append(cmd)
        pass    

    def __repr__(self):
        return "\n".join(self.cmds)
        
    def __call__(self):
        n = len(self.cmds)
        for i, cmd in enumerate(self.cmds):
            if i % 100 == 0:
                log.info("[%0.3d/%0.3d] %s " % ( i+1, n, cmd))
            pass
            for line in os.popen(cmd).readlines():
                log.info("    %s " % line.strip())
            pass



def read_cnf_( path ):
    try:
        from ConfigParser import SafeConfigParser
    except ImportError:
        from configparser import ConfigParser as SafeConfigParser
    pass
    path = os.path.expanduser(path)
    assert os.path.exists(path), path
    log.debug("reading %s " % ( path ) )
    cnf = SafeConfigParser()
    cnf.read(path)
    return cnf

def parse_args_(doc, **kwa):

    d = {}
    d["cnfpath"] = "~/.workflow.cnf"
    d["cnfsect"] = "html2wdocs"
    d["logformat"] = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    d["loglevel"] = "INFO"
    d.update(kwa)

    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-c", "--cnfpath", default=d["cnfpath"], help="Path to config file, default %default, that holds URLs and access keys for the API" )
    op.add_option("-s", "--cnfsect", default=d["cnfsect"], help="Section of config file to read, default %default " )
    op.add_option("-g", "--logpath", default=None )
    op.add_option(      "--PROCEED", action="store_true", default=False, help="Proceed to run the commands, default %default " )
    op.add_option("-t", "--logformat", default=d["logformat"] )
    op.add_option("-l", "--loglevel", default=d["loglevel"], help=" default %default " )
    opts, args = op.parse_args()
    opts.cnf = read_cnf_( opts.cnfpath )

    level=getattr(logging,opts.loglevel.upper()) 
    if opts.logpath:
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
    else:
        logging.basicConfig(format=opts.logformat,level=level)

    return opts, args

def main(**kwa):
    opts, args = parse_args_(__doc__, **kwa)
    cfg = dict(opts.cnf.items(opts.cnfsect))
    argv0 = os.path.basename(sys.argv[0])
    assert argv0 == cfg['argv0'], ( argv0, cfg, "config argv0 mismatch with script" )
    log.info(cfg) 
    pz = Parallelize( cfg['source'], cfg['target'] )     

    if opts.PROCEED:
       log.warn("proceeding")
       pz()
    else:
       print(pz)
       log.warn("run again with --PROCEED to do this commands")


if __name__ == '__main__':
    main()
   


