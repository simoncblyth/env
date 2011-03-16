#!/usr/bin/env python
"""
http://jtauber.com/python_subversion_binding/

Dev is not very convenient as to see whats really 
happening have to use as real hook ... forcing non-interactive

Failed to workaround by using -testhook 
to guess the next txn_name then use that with -ihook   
... as that gives not such txn 


Approaches to validiation candidate DBI updates 

  * parse svnlook diff output 

  * get to grips with python svn api and apply difflib to the 
    before and after file versions ... if that is possible
    ... THIS AVOIDS SOME TRICKSY PARSING AT COST OF LESS PORTABILITY
    ... BUT ONLY NEEDS TO RUN ON SERVER 

     ... NOT TRUE WANT A CLIENT VERSION THAT CAN BE APPLIED TO AN 
           svn diff  outpiut also  

"""
import os, sys
from svn import fs, repos, core
assert (core.SVN_VER_MAJOR, core.SVN_VER_MINOR) >= (1, 3), "Subversion 1.3 or later required"

log = lambda m:sys.stderr.write(m)


def get_new_paths(txn_root):
    new_paths = []
    for path, change in fs.paths_changed(txn_root).iteritems():
        if (change.change_kind == fs.path_change_add or change.change_kind == fs.path_change_replace):
            new_paths.append(path)
    return new_paths

def list_paths(txn_root):
    for path, change in fs.paths_changed(txn_root).iteritems():
       log("list_paths %r %r " % ( path, change) ) 


def precommit_svn_api(repo_path, txn_name):
    log( "repo_path %s txn_name %s \n" % ( repo_path, txn_name ) )
    repository = repos.open(repo_path)
    fs_ptr = repos.fs(repository)
    log("%r\n" % repository ) 
    log("%r\n" % fs_ptr ) 
    rev = fs.youngest_rev(fs_ptr)
    log("youngest %r\n" % rev ) 
    txn_ptr = fs.open_txn(fs_ptr, txn_name)
    log("txn_ptr %r\n" % txn_ptr ) 
    txn_root = fs.txn_root(txn_ptr)
    log("txn_root %r\n" % txn_root ) 
    list_paths(txn_root)
    rc = 1
    return locals() 


class SVNLook(dict):
    _cmd = "%(exepath)s %(cmd)s %(repo_path)s --transaction %(txn_name)s "
    cmd = property(lambda self:self._cmd % self ) 
    def __call__(self, *args, **kwargs):
        self.update( kwargs )
        log("%r"%self)
        return os.popen( self.cmd ).read()


def precommit_svnlook(repo_path, txn_name ):
    svnlook = os.environ.get("SVNLOOK")
    slk = SVNLook( txn_name=txn_name, repo_path=repo_path, exepath=os.environ.get('SVNLOOK') )

    log(slk(cmd="log"))
    log(slk(cmd="author"))
    log(slk(cmd="diff"))

    rc = 1
    return locals()

if __name__=='__main__':
    ## kludge to allow interactive dev and yet be a real hook too
    locals().update( precommit_svnlook(sys.argv[1],sys.argv[2]) )
    try:
        __IPYTHON__
    except NameError:
        sys.exit(rc)






