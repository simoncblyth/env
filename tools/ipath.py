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
import os, hashlib, logging, re, sys, commands
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


class IPath(object):
    """
    Holder object for:

    #. path
    #. content digest
    #. svn status string

    """

    @classmethod
    def repotype(cls, _, up=0):
        """
        :param _: absolute path 
        :param up: number of directories up from original
        :return typ, base:  repo type and base directory  

        Identity repository type as svn/git/hg for path argument 
        and provide base directory of the repo.
        """
        log.debug("repotype %2d %s " % (up, _) )

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
            if typ is None and up < 10 and os.path.dirname(_) != _:
                typ, base = cls.repotype( os.path.dirname(_), up=up+1 ) 
            pass
        else:
            log.debug("not file or dir [%s] eg when non-existing target path  " % _ )
            typ, base = None, None
        pass
        return typ, base

    @classmethod
    def status_command(cls, _, typ, base=None):
        """
        :param _: directory path
        :return: status command for the repository 

        """
        if typ == "svn":
            cmd = "svn status %s " % _
        elif typ == "git":
            if _ == base:
                _ = ""
            pass
            cmd = "git --work-tree=%(base)s --git-dir=%(base)s/.git status --porcelain %(_)s " % locals()
            pass
        elif typ == "hg":
            cmd = "hg status %s " % _
        else:
            assert 0, typ
        pass
        return cmd

    @classmethod
    def run(cls, cmd):
        rc, out = commands.getstatusoutput(cmd)
        if rc != 0:
            log.fatal("non-zero RC from cmd : %s " % cmd ) 
        pass     
        assert rc == 0 
        log.debug("cmd:[%s] out:%d " % (cmd, len(out)) ) 
        return out 

    @classmethod
    def parse(cls, line, ptn):
        m = ptn.match(line)
        if m:
            groups = m.groups()
            assert len(groups) == 3 
            stat_,atat_,mpath = groups
            stat = stat_.rstrip()
            log.debug( "[%s][%s] %s " % ( stat, atat_, mpath ))
        else:
            log.debug("no match [%s] " % line )   
            stat,mpath = "_", None
        pass
        return stat, mpath

    @classmethod
    def multiparse(cls, lines, ptn, pfx):
        sub = []
        log.debug("multiparse %d lines " % len(lines) ) 
        for line in lines:
            stat, mpath = cls.parse(line, ptn)
            path = mpath if pfx is None else os.path.join(pfx, mpath) 
            s = cls( path, stat=stat )
            sub.append( s ) 
        pass
        return sub

    def __init__(self, path_, stat=None):
        """
        :param path: 
        """
        xx_ = lambda _:os.path.abspath(os.path.expandvars(os.path.expanduser(_)))
        path = xx_(path_)
        typ, base = self.repotype(path)
        dig = digest_(path)
        isdir = os.path.isdir(path)

        cwd = os.getcwd()
        cpath = path[len(cwd)+1:]   # relative to cwd
    

        cmd, out, ptn, rpath, sub = None, None, None, None, []

        if typ is not None and stat is None:
            rpath = path[len(base)+1:]   # relative to repo base
            rbase = base[len(cwd)+1:]   # base dir relative to cwd 

            cmd = self.status_command( cpath, typ, rbase)   
            out = self.run(cmd) 

            # git uses relative to base paths in status, hg/svn use relative to cwd
            upath = rpath if typ == "git" else cpath

            ptns = "^\s*(\S)(\S?)\s*(%s.*)\s*$" % upath  
            log.debug("ptns:[%s]" % ptns )

            ptn = re.compile(ptns)
     
            pfx = rbase if typ == "git" else None

            lines = out.split("\n")
            if len(lines) == 1:
                stat, mpath = self.parse(lines[0], ptn)
            else:
                sub = self.multiparse(lines, ptn, pfx=pfx )
                pass
            pass
        pass
        self.path = path 
        self.rpath = rpath 
        self.typ = typ
        self.base = base
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
    logging.basicConfig(level=logging.INFO)
    assert len(sys.argv) > 1      

    path = os.path.abspath(sys.argv[1])
 
    typ = IPath.repotype(path)
    print typ

    p = IPath(path)
    print repr(p) 

    for s in p.sub:
        print repr(s) 



    
