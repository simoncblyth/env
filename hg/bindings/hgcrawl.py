#!/usr/bin/env python
"""
::

    hgcrawl.py /tmp/mercurial/env -r300

    hgcrawl.py /tmp/mercurial/env -r300 -m/
       #
       # dump the contents digest for all paths at the specified revision 


"""
import os, argparse, logging, hashlib
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


def mimic_svn_link_digest(link):
    assert os.path.islink(link)
    target = os.readlink(link)
    mimic = "link %s" % target 
    digest = hashlib.md5(mimic).hexdigest() 
    log.info("link %s target %s mimic %s digest %s " % (link,target,mimic,digest)) 
    return digest 


class HGCrawler(object):
    def __init__(self, hgdir, verbose=False, exclude_dirs=[".hg"]):
        assert os.path.exists(os.path.join(hgdir,'.hg'))
        self.verbose = verbose
        self.hg = hgapi.Repo(hgdir)
        self.hgdir = hgdir
        self.exclude_dirs = exclude_dirs
        self.rootpath = hgdir

    def reset(self, revision=None):
        self.revision = revision
        self.dirs = []
        self.paths = []

    def recurse(self, hgrev):
        """
        :param hgrev:

        Upadates Mercurial working copy to the revision specified
        and recursively crawls the filesystem
        """
        self.reset(hgrev)
        self.hg.hg_update(hgrev)
        crawler( self.hgdir, self.hgdir, self.dir_, self.leaf_, self.rootpath, self.exclude_dirs )

    def contents_digest(self):
        pass
        def _digest(fp):
            md5 = hashlib.md5()
            for chunk in iter(lambda: fp.read(8192),''): 
                md5.update(chunk)
            return md5.hexdigest()
        pass
        def _resolve(p):
            if p[0] == '/':
                p = p[1:]
            path = os.path.join(self.hgdir, p)
            log.debug("hgdir %s p %s path %s " % (self.hgdir, p, path ))
            return path
        pass
        digest = {}
        for p in self.paths:
            path = _resolve(p)
            if os.path.islink(path):
                digest[p] = mimic_svn_link_digest( path )
            else:
                with open(path,"rb") as fp:
                    digest[p] = _digest(fp)
            pass 
        return digest

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
    parser.add_argument( "-m","--md5", default=None )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging,args.loglevel.upper()))
    return args 
    
def main():
    args = parse(__doc__)
    hc = HGCrawler(args.path[0], verbose=args.verbose)
    hc.recurse(args.revision)

    path = args.md5
    if not path is None:
        digest = hc.contents_digest() 
        if path in digest:
            print digest[path]
        else:
            print digest
        pass 
    pass

    print hc 




if __name__ == '__main__':
    main()


