#!/usr/bin/env python
"""
::

    hgcrawl.py /tmp/mercurial/env -r300

"""
import os, argparse, logging
log = logging.getLogger(__name__)
import hgapi


def crawler(root, directory, dir_, leaf_, rootpath, exclude_dirs=[".hg"]):
    """
    :param root: root directory that passes along unchanged through the recursion
    :param directory: absolute directory inside the root

    Recursive crawler
    """
    dir_(directory[len(rootpath):])
    assert directory.startswith(root)
    for name in os.listdir(directory):
        path = os.path.join(directory,name)
        if os.path.isdir(path):
            if not name in exclude_dirs:
                crawler( root, path, dir_, leaf_, rootpath, exclude_dirs )
            pass
        else:
            leaf_(path[len(rootpath):])
        pass


class HGCrawler(object):
    def __init__(self, hgdir, verbose=False, exclude_dirs=[".hg"]):
        self.verbose = verbose
        self.hg = hgapi.Repo(hgdir)
        self.hgdir = hgdir
        self.exclude_dirs = exclude_dirs
        self.rootpath = hgdir

    def reset(self, revision=None):
        self.revision = revision
        self.dirs = []
        self.paths = []

    def __call__(self, hgrev):
        """
        :param hgrev:

        Upadates Mercurial working copy to the revision specified
        and recursively crawls the filesystem
        """
        self.reset(hgrev)
        self.hg.hg_update(hgrev)
        crawler( self.hgdir, self.hgdir, self.dir_, self.leaf_, self.rootpath, self.exclude_dirs )

    def dir_(self, rpath):
        self.dirs.append(rpath)
        if self.verbose:
            print path + '/'

    def leaf_(self, rpath):
        self.paths.append(rpath)
        if self.verbose:
            print rpath
  
    def __repr__(self):
        return "%s %s %s dirs %s leaves %s " % (self.__class__.__name__, self.hgdir, self.revision,len(self.dirs),len(self.paths))





def parse(doc):
    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-v","--verbose", action="store_true")
    parser.add_argument(     "--loglevel", default="info")
    parser.add_argument( "path", nargs=1 )
    parser.add_argument( "-r","--revision", default=None )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging,args.loglevel.upper()))
    return args 
    
def main():
    args = parse(__doc__)
    hc = HGCrawler(args.path[0], verbose=args.verbose)
    hc(args.revision)
    print hc 




if __name__ == '__main__':
    main()


