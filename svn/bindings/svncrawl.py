#!/usr/bin/python
"""

Crawl a repository, printing versioned object path names.::

    svncrawl.py -r1000 /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637


Based on example from 

* http://svnbook.red-bean.com/en/1.7/svn.developer.usingapi.html

"""
import sys, argparse, logging
import os.path
import svn.fs, svn.core, svn.repos

log = logging.getLogger(__name__)

def crawl_filesystem_dir(root, directory, dir_, leaf_, exclude_dirs=['tags','branches'], rootpath='/trunk', skipempty=False):
    """
    Recursive crawler of SVN repository 
    """
    entries = svn.fs.svn_fs_dir_entries(root, directory)
    names = entries.keys()
    reldir = directory[len(rootpath):]  
    if skipempty and len(names) == 0:
        log.info("skipempty dir %s " % reldir )
    else:
        dir_(reldir)
    pass

    for name in names:
        full_path = directory + '/' + name
        if svn.fs.svn_fs_is_dir(root, full_path):
            if not name in exclude_dirs:
                crawl_filesystem_dir(root, full_path, dir_, leaf_, exclude_dirs, rootpath, skipempty)
        else:
            leaf_(full_path[len(rootpath):])
        pass
    pass


class SVNCrawler(object):
    def __init__(self, repos_path, exclude_dirs=['tags','branches'], rootpath="/trunk", verbose=False, skipempty=False):
        """
        :param repos_path:
        :param revision:
        """
        self.repos_path = repos_path
        self.exclude_dirs = exclude_dirs
        self.rootpath = rootpath
        self.verbose = verbose
        self.skipempty = skipempty
        self.reset()

    def reset(self, revision=None):
        self.revision = revision
        self.paths = []
        self.dirs = []

    def __call__(self, revision=None):
        """
        Passing root_obj between scopes causes segmentation faults, 
        SWIG can be that way.
        """
        repos_obj = svn.repos.svn_repos_open(self.repos_path)
        fs_obj = svn.repos.svn_repos_fs(repos_obj)
        rev = svn.fs.svn_fs_youngest_rev(fs_obj) if revision is None else int(revision)
        root_obj = svn.fs.svn_fs_revision_root(fs_obj, rev)
 
        self.reset(revision)
        crawl_filesystem_dir(root_obj, self.rootpath, self.dir_, self.leaf_, self.exclude_dirs, self.rootpath, self.skipempty)
         
    def leaf_(self, rpath):
        self.paths.append(rpath)
        if self.verbose:
            print rpath

    def dir_(self, rpath):
        self.dirs.append(rpath)
        if self.verbose:
            print rpath + '/'

    def __repr__(self):
        return "%s %s %s dirs %s leaves %s " % (self.__class__.__name__, self.repos_path, self.revision,len(self.dirs),len(self.paths))

def parse(doc):
    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-v","--verbose", action="store_true")
    parser.add_argument(     "--skipempty", action="store_true")
    parser.add_argument(     "--loglevel", default="info")
    parser.add_argument( "path", nargs=1 )
    parser.add_argument( "-r","--revision", default=None )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging,args.loglevel.upper()))
    return args 
    
def main():
    args = parse(__doc__)
    repos_path = svn.core.svn_dirent_canonicalize(args.path[0])
    log.debug("repos_path %s " % (repos_path))
    sc = SVNCrawler(repos_path, verbose=args.verbose, skipempty=args.skipempty)
    sc(args.revision)
    print sc


if __name__ == "__main__":
    main()




