#!/usr/bin/env python
"""
Using subversion swig python bindings to allow
programmatic checking of a "hg convert" migration 
frob SVN to Mercurial. 


"""
import os, logging
import svn
import svn.repos

log = logging.getLogger(__name__)


class SVNRepo(object):
    def __init__(self, path):
        repo_ptr = svn.repos.open(path)
        repo_ptr.assert_valid()
        fs_ptr = svn.repos.fs(repo_ptr)
        self.repo_ptr = repo_ptr
        self.fs_ptr = fs_ptr

    youngest_rev = property(lambda self:svn.fs.youngest_rev(self.fs_ptr))



def main():
    logging.basicConfig(level=logging.INFO)
    svndir = os.environ['SCMMIGRATE_SVNDIR']
    r = SVNRepo(svndir)
    log.info("svndir %s youngest_rev %s " % (svndir, r.youngest_rev ))


if __name__ == '__main__':
    main()
    



