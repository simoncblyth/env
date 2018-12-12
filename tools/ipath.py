#!/usr/bin/env python
"""
ipath.py : file system path instrumented with svn/hg/git repo metadata
========================================================================== 

Testing with svn, git and hg repos::

    delta:~ blyth$ env/tools/ipath.py workflow
    delta:~ blyth$ env/tools/ipath.py assimp-fork
    delta:~ blyth$ env/tools/ipath.py env

"""
from __future__ import with_statement
import os, hashlib, logging, re, sys, subprocess
from functools import partial

try:
    import commands
except ImportError:
    commands = None
pass


log = logging.getLogger(__name__)

def old_digest_(path):
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
        pass
    pass
    return md5.hexdigest()

def digest_(path):
    """
    https://stackoverflow.com/questions/7829499/using-hashlib-to-compute-md5-digest-of-a-file-in-python-3
    """
    if not os.path.exists(path):return None
    if os.path.isdir(path):return None
    d = hashlib.md5()
    with open(path, mode='rb') as f:
        for buf in iter(partial(f.read, 8192), b''):
            d.update(buf)
        pass
    pass
    return d.hexdigest()



class IPath(object):
    """
    Holder object for:

    #. path
    #. content digest
    #. svn status string

    """

    @classmethod
    def repotype(cls, _, up=0, noup=False):
        """
        :param _: absolute path 
        :param up: number of directories up from original
        :return typ, base:  repo type and base directory  

        Identity repository type as svn/git/hg for path argument 
        and provide base directory of the repo.   This works by recursively 
        moving up the directory tree looking for special folders .hg .svn .git 
        """
        #log.debug("repotype %2d %s " % (up, _) )

        if os.path.isfile(_):
            typ, base = cls.repotype( os.path.dirname(_) )
        elif os.path.isdir(_):
            if os.path.isdir(os.path.join(_, ".hg")):
                typ, base = "hg", _
            elif os.path.isdir(os.path.join(_, ".svn")):
                typ, base = "svn", _
            elif os.path.isdir(os.path.join(_, ".git")):
                typ, base = "git", _
            else:
                typ, base = None, None
            pass
            if noup == False and typ is None and up < 10 and os.path.dirname(_) != _:
                typ, base = cls.repotype( os.path.dirname(_), up=up+1 ) 
            pass
        else:
            log.debug("not file or dir [%s] eg when non-existing target path  " % _ )
            typ, base = None, None
        pass
        return typ, base

    @classmethod
    def status_command(cls, _, typ, rbase=None):
        """
        :param _: directory path
        :param typ: repo type
        :param rbase: repodir expressed relative to cwd  
        :return: status command for the repository 

        """
        if typ == "svn":
            cmd = "svn status %s " % _
        elif typ == "git":
            if len(rbase) == 0:  ## cwd is the repodir
                cmd = "git status --porcelain %(_)s " % locals()
            else:
                cmd = "git --work-tree=%(rbase)s --git-dir=%(rbase)s/.git status --porcelain %(_)s " % locals()
            pass
        elif typ == "hg":
            cmd = "hg status %s " % _
        else:
            assert 0, typ
        pass
        return cmd

    @classmethod
    def run(cls, cmd):

        if not commands is None:
            rc, out = commands.getstatusoutput(cmd)
        else:
            #cpr = subprocess.run(cmd.split(" "), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            args = list(filter(None, cmd.split(" ")))  
            ## mysteriously leaving a space on the end, causes to list many files, not just the one
            ##print(args)

            cpr = subprocess.run(args, check=True, text=True, capture_output=True)
            out = cpr.stdout
            rc = cpr.returncode 
            ##print(str(out))
        pass
        if rc != 0:
            log.fatal("non-zero RC from cmd : %s " % cmd ) 
        pass     
        assert rc == 0,  rc
        log.debug("cmd:[%s] out:%d " % (cmd, len(out)) ) 
        return out 

    @classmethod
    def parse(cls, line, ptn):
        log.debug("parse line [%s] " % line )
        m = ptn.match(line)
        if m:
            groups = m.groups()
            assert len(groups) == 3 
            stat_,atat_,mpath = groups
            stat = stat_.rstrip()
            log.debug( "match [%s][%s] %s " % ( stat, atat_, mpath ))
        else:
            log.debug("no match [%s] " % line )   
            assert 0
            stat,mpath = "_", None
        pass
        #log.info("parse line %s DONE " % line )
        return stat, mpath

    @classmethod
    def multiparse(cls, lines, ptn, pfx):
        sub = []
        log.debug("multiparse %d lines " % len(lines) ) 
        for line in filter(None, lines):
            stat, mpath = cls.parse(line, ptn)
            if mpath is None:
                log.warning("skipping unmatched line [%s]" % line)
                continue
            pass
            path = mpath if pfx is None else os.path.join(pfx, mpath) 
            s = cls( path, stat=stat )
            sub.append( s ) 
        pass
        log.debug("multiparse %d lines len(sub) %d DONE " % (len(lines), len(sub)) ) 
        pass
        return sub

    def __init__(self, path_, stat=None, noup=False):
        """
        :param path: relative or absolute, tilde or envvars are expanded
        """
        log.debug("IPath.path_ %s " % path_ )
        xx_ = lambda _:os.path.abspath(os.path.expandvars(os.path.expanduser(_)))
        path = xx_(path_)
        typ, repodir = self.repotype(path, noup=noup)  
        log.debug(" HERE path_:%s typ:%s repodir:%s " % (path_, typ, repodir))
        reponame = os.path.basename(repodir) if repodir is not None else None 

        log.debug(" IPath path_:%s path:%s typ:%s repodir:%s reponame:%s " % ( path_, path, typ, repodir, reponame) )

        dig = digest_(path)
        isdir = os.path.isdir(path)

        cwd = os.getcwd()
        cpath = path[len(cwd)+1:]   # path relative to cwd
      
        cmd, out, ptn, rpath, sub = None, None, None, None, []

        if typ is not None and stat is None:
            rpath = path[len(repodir)+1:]   # path relative to repodir
            rbase = repodir[len(cwd)+1:]    # repodir expressed relative to cwd 

            cmd = self.status_command( cpath, typ, rbase)   
            log.debug("cpath:%s rpath:%s rbase:%s " % (cpath, rpath, rbase))
            log.debug("cmd:%s " % (cmd))

            out = self.run(cmd) 
            #print(out)

            # git uses relative to base paths in status, hg/svn use relative to cwd
            upath = rpath if typ == "git" else cpath

            ptns = "^\s*(\S)(\S?)\s*(%s.*)\s*$" % upath  
            log.debug("ptns:[%s]" % ptns )

            ptn = re.compile(ptns)
     
            pfx = rbase if typ == "git" else None

            lines = list(filter(None, out.split("\n")))

            nline = len(lines)
            log.debug("read %d lines from cmd:%s" % (nline, cmd) )
            if not isdir:
                assert nline <= 1 
            pass
            ## hmm : problem for a single binary to be sweeped
            ## single line is degenerate between a directory with a single item
            ## and the IPath of an item in the dir   
            ## if nline == 1:
            ##    stat, mpath = self.parse(lines[0], ptn)
            ##else:
            ##    sub = self.multiparse(lines, ptn, pfx=pfx )
            ##   pass
            ##pass
            if isdir:
                sub = self.multiparse(lines, ptn, pfx=pfx )
            else:
                if len(lines) > 0:
                    stat, mpath = self.parse(lines[0], ptn)
                else:
                    pass
                pass
            pass
        pass
        self.path = path 
        self.rpath = rpath 
        self.typ = typ
        self.repodir = repodir
        self.reponame = reponame
        self.digest = dig
        self.isdir = isdir
        self.cmd = cmd 
        self.out = out
        self.stat = stat 
        self.is_untracked = stat in ["??", "?"] 
        self.sub = sub
        pass
    def __repr__(self):
        return "[%s] %s %s DIR:%s " % ( self.stat if self.stat is not None else "_", self.path, self.digest, self.isdir )     

    def __str__(self):
        return "[%s] %s %s DIR:%s out:%s cmd:%s " % ( self.stat, self.path, self.digest, self.isdir, self.out, self.cmd )     





if __name__ == '__main__':

    level = "INFO"
    #level = "DEBUG"

    logging.basicConfig(level=getattr(logging, level))
    assert len(sys.argv) > 1      

    path = os.path.abspath(sys.argv[1])
 
    typ, base = IPath.repotype(path)
    log.info("typ:%s base:%s " % (typ,base) )

    p = IPath(path)
    log.info("p:%r " % (p,) )

    for s in p.sub:
        log.info("s:%r " % (s,) ) 



    
