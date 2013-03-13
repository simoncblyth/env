#!/usr/bin/env python
"""
Usage::

    python tools/parallizer.py -h
    python tools/parallizer.py        # check 
    python tools/parallizer.py | sh   # do


"""
from __future__ import with_statement
import os, sys, hashlib, logging, re
log = logging.getLogger(__name__)

def digest_(path):
    """
    :param path:
    :return: md5 hexdigest of the content of the path or None if non-existing path

    http://stackoverflow.com/questions/1131220/get-md5-hash-of-a-files-without-open-it-in-python

    Confirmed to give same hexdigest as commandline /sbin/md5::

        md5 /Users/blyth/workflow/notes/php/property/colliers-4q2011.pdf 
        MD5 (/Users/blyth/workflow/notes/php/property/colliers-4q2011.pdf) = 3a63b5232ff7cb6fa6a7c241050ceeed

    """
    if not os.path.exists(path):return None
    if os.path.isdir(path):return None
    md5 = hashlib.md5()
    with open(path,'rb') as f: 
        for chunk in iter(lambda: f.read(8192),''): 
            md5.update(chunk)
    return md5.hexdigest()


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
        

def read_cnf_( path ):
    from ConfigParser import SafeConfigParser
    path = os.path.expanduser(path)
    assert os.path.exists(path), path
    log.debug("reading %s " % ( path ) )
    cnf = SafeConfigParser()
    cnf.read(path)
    return cnf

def parse_args_(doc):
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-c", "--cnfpath", default="~/.workflow.cnf", help="Path to config file that holds URLs and access keys for the API" )
    op.add_option("-s", "--cnfsect", default="html2wdocs", help="Section of config file to read" )
    op.add_option("-g", "--logpath", default=None )
    op.add_option("-t", "--logformat", default="%(asctime)s %(name)s %(levelname)-8s %(message)s" )
    op.add_option("-l", "--loglevel", default="INFO", help=" default %default " )
    opts, args = op.parse_args()
    opts.cnf = read_cnf_( opts.cnfpath )

    level=getattr(logging,opts.loglevel.upper()) 
    if opts.logpath:
        logging.basicConfig(format=opts.logformat,level=level,filename=opts.logpath)
    else:
        logging.basicConfig(format=opts.logformat,level=level)

    return opts, args

def main():
    opts, args = parse_args_(__doc__)
    cfg = dict(opts.cnf.items(opts.cnfsect))
    argv0 = os.path.basename(sys.argv[0])
    assert argv0 == cfg['argv0'], ( argv0, cfg, "config argv0 mismatch with script" )
    log.info(cfg) 
    pz = Parallelize( cfg['source'], cfg['target'] )     
    print pz

if __name__ == '__main__':
    main()
   


