#!/usr/bin/env python
"""
SWEEPS SVN WORKING COPY INTO INTO PARALLEL DIRECTORY STRUCTURE
================================================================

Typical usage is to get PDF/PNGs etc.. and other binaries out 
of a source tree and into a parallel hierarcy, typically 
for local webserver presentation from there.

The veracity of the copy is checked via digests.  

.. note:: the advantage is organization within single tree heirarchy tree, without clutering repo with binaries

.. warn:: deletes questionmarked directories from working copy, so cleanup svn status manually before using this to complete the task
          potentially loosing files if their extensions do not correspond to the selected ones



Specifying exts changes behaviour to less efficient dir_walk 
and sweeping no-matter what the svn status 
ie here all .txt not already sweeped will be sweeped



Avoid copying around trash like .aux .log and _build

#. setting svn ignores to skip them from status, eg  

   * globally in `~/.subversion/config` 
   * locally with `svn pe svn:ignore .` in the directory

Avoid folder deletions by

#. adding them to repo 
#. setting global or local svn ignores to avoid "?" svn status


Issues
~~~~~~~~

#. .numbers and .key "files" are seen as directories and thus skipped from the sweep

Workaround is to sweep/check/clean manually::

    simon:workflow blyth$ cp -R admin/address/Domicile.key ~/WDOCS/admin/address/
    simon:workflow blyth$ open ~/WDOCS/admin/address/Domicile.key # check it is OK
    simon:workflow blyth$ rm -rf admin/address/Domicile.key




This script just looks and suggests the commands to run, it does not do anything.  
To act on suggestions pipe the output to the shell.

Although Sphinx has a download directive, it works via duplication

Usage, takes 2 runs to 1st copy binaries from working copy and then to verify them and delete in working copy

   cd ~/workflow
   ./sweep.py           ## check the copies/deletions to be done
   ./sweep.py | sh     ## do them
   ./sweep.py           ## check the copies/deletions to be done
   ./sweep.py | sh     ## do them 

   cd ~/env
   
Note that this module is imported by these lightweight sweep scripts that live
in the root of each svn repository.

Refer to resources with urls like:

  * http://localhost/edocs/whatever.pdf    
  * http://localhost/wdocs/notes/dev/whatever.pdf    
  * OR :w:`notes/dev/whatever.pdf`

TODO:

#. hmm, I now want the working copy to live on G4PB for easy access to notes : need to change names from workflow to wdocs
#. could base off the ``svn status`` output rather than walking and calling ``svn status`` for each 

   #. ` svn st | grep ^\?`


"""
from __future__ import with_statement
import os, hashlib, logging, re, sys
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

def svnstatus_(path):
    """
    :param path:
    :return: svn status string of the path or "-" if blank 
    """
    st = os.popen("svn status %s 2>/dev/null" % path).read()
    if len(st) == 0:
        return "-"
    else:
        return st[0]

class DPath(object):
    """
    Holder object for:

    #. path
    #. content digest
    #. svn status string

    """
    def __init__(self, path, svnst=None):
        """
        :param path: 
        :param svnst: status string
        """
        self.path = path 
        self.digest = digest_(path)
        self.isdir = os.path.isdir(path)
        if svnst:
            self.svnst = svnst
        else:    
            self.svnst = svnstatus_(path)
    def __repr__(self):
        return "[%s] %s %s DIR:%s " % ( self.svnst, self.path, self.digest, self.isdir )     


class Sweeper(list):
    """
    Keep working copy clean by sweeping chosen filetypes into a 
    parallel heirarchy.

    Deficiencies:

    #. probably will not handle major changes in working copy structure very gracefully

       * eg does not delete at target on changing source directory names, so orphans will result

    """
    def __init__(self, src, tgt, exts=None , skipd=None ):    
        """
        :param src: source SVN working copy directory 
        :param tgt: target directory
        :param exts: default of None sweeps all working copy detritus with svn status of `?`

        if a whitespace delimited string is provided such as ".txt .rst" then a
        simple filesystem walk is used rather than svn status running

        :param skip: space delimited list of dirnames to skip in the dir_walk (ignored in status_walk) eg "_build _sources"

        """
        xx_ = lambda _:os.path.expandvars(os.path.expanduser(_))
        self.src = xx_(src)
        self.tgt = xx_(tgt)

        self.cmds = []
        self.skipd = skipd.split() if skipd else []
        if not exts:
            self.exts = None
            self.status_walk()
        else:
            self.exts = exts.split()
            self.dir_walk()

    def status_walk(self):
        """
        Pattern match the `svn status` of the `src` directory  passing all matches to `handle`
        Note that only a single `svn status` is required, so much more efficient compared to oldwalk.
        """
        cmd = "svn status %s " % self.src
        ptn = re.compile("^(\S)\s*(\S?)\s*(%s.*)\s*$" % self.src)
        for line in os.popen(cmd).readlines():
            m = ptn.match(line)
            if m:
                groups = m.groups()
                assert len(groups) == 3 
                stat_,atat_,path = groups
                stat = stat_.rstrip()
                log.debug( "[%s][%s] %s " % ( stat, atat_, path ))
                self.clean_handle( path, stat )
            else:
                log.info("no match %s " % line )   

    def dir_walk(self):
        """
        Simple os.walk the source tree handling files with extensions matching the 
        selected ones.
        """
        for dirpath, dirs, names in os.walk(self.src):
            rdir = dirpath[len(self.src)+1:]    
            for skp in self.skipd:         
                if skp in dirs:
                    dirs.remove(skp)  
            for name in names:
                root, ext = os.path.splitext(name)
                if not ext in self.exts: 
                    continue
                spath = os.path.join(dirpath, name)
                self.copy_handle(spath)

    def copy_handle(self, spath):
        """
        :param spath: full path 
        """ 
        rpath = spath[len(self.src)+1:]
        sp = DPath(spath, svnst="-") 
        tpath = os.path.join(self.tgt,rpath)
        tp = DPath(tpath, svnst="-") 
        if sp.digest == tp.digest:
            log.debug("no update needed %r " % sp )         
        else:   
            log.info("sp %s " % ( sp ))
            log.info("tp %s " % ( tp ))
            cmd = "mkdir -p \"%s\" && cp \"%s\" \"%s\" " % ( os.path.dirname(tpath), spath, tpath )
            self.cmds.append(cmd)

    def clean_handle(self, spath, svnst=None):
        """
        Considers a source path and appends shell commands to either:

        #. create target directory if not created and copy the binary littering source to target
        #. remove the littering binary from working copy source if digests of verify a good copy to target 

        :param spath: full path 
        :param svnst: svn status string
        """
        rpath = spath[len(self.src)+1:]

        sp = DPath(spath, svnst=svnst) 
        if sp.svnst != '?':
            return 

        if sp.isdir:
            log.warn("skipping uncommitted directory : %r " % sp )    
            return         

        tpath = os.path.join(self.tgt,rpath)
        tp = DPath(tpath, svnst="-") 
        log.info("sp %s" % sp )
        log.info("tp %s" % tp )

        if sp.digest == tp.digest:
            cmd = "rm -f \"%s\" " % ( spath )         
        else:   
            cmd = "mkdir -p \"%s\" && cp \"%s\" \"%s\" " % ( os.path.dirname(tpath), spath, tpath )
        self.cmds.append(cmd)

    def __repr__(self):
        return "\n".join(self.cmds)
        


def parse_args_(doc):
    from optparse import OptionParser
    op = OptionParser(usage=doc)
    op.add_option("-c", "--cnfpath", default=os.environ['SVNSWEEP_CNFPATH'], help="Path to config file, default %default " )
    op.add_option("-s", "--cnfsect", default=os.environ['SVNSWEEP_CNFSECT'], help="Comma delimeted list of config file sections to read, default %default " )
    op.add_option("-g", "--logpath", default=None )
    op.add_option(      "--PROCEED", action="store_true", default=False, help="Proceed to run the commands, default %default " )
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


def read_cnf_( path ):
    from ConfigParser import SafeConfigParser
    path = os.path.expanduser(path)
    assert os.path.exists(path), path
    log.debug("reading %s " % ( path ) )
    cnf = SafeConfigParser()
    cnf.read(path)
    return cnf


def main():
    opts, args = parse_args_(__doc__)

    for sect in opts.cnfsect.split(","):
        cfg = dict(opts.cnf.items(sect))
        argv0 = os.path.basename(sys.argv[0])
        assert argv0 == cfg['argv0'], ( argv0, cfg, "config argv0 mismatch with script" )
        log.info(cfg)
        swp = Sweeper( cfg['source'], cfg['target'], exts=cfg.get('exts',None), skipd=cfg.get('skipd',None) )     
        print swp



if __name__ == '__main__':
    main()


