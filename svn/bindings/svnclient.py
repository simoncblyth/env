#!/usr/bin/env python
"""
SVNCLIENT.PY
==============

Control of SVN working copy via pysvn client.

#. trunk doesnt exist at revision 0
#. an alternative approach using direct access to SVN repo database 
   is in svncrawl.py 


"""
import os, argparse, logging, hashlib
import IPython as IP
from datetime import datetime
import pysvn
from env.scm.timezone import cst, utc

from svncommon import unprefix, mimic_svn_link_digest


log = logging.getLogger(__name__)
rev_ = lambda _:pysvn.Revision( pysvn.opt_revision_kind.number, _ )




def crawler(root, directory, dir_, exclude_dirs=[".svn"], skipempty=False ):
    """
    Typically start recursion with `root` and `directory` the same

    :param root: root directory that passes along unchanged through the recursion
    :param directory: absolute directory inside the root, can also be the root
    :param dir_:
    :param exclude_dirs: 

    Recursive crawler.  
    """
    assert directory.startswith(root)
    rrdir = directory[len(root):]  

    leaves = []
    for name in os.listdir(directory):
        path = os.path.join(directory,name)
        if os.path.isdir(path):
            if not name in exclude_dirs:
                subleaves = crawler( root, path, dir_, exclude_dirs, skipempty  )
                leaves.extend(subleaves)
        else:
            leaves.append(path[len(root):])
        pass
    pass
    if len(leaves) == 0 and skipempty:
        log.debug("skipempty rrdir %s " % rrdir )
    else:   
        dir_(rrdir)
    pass
    return leaves


class SVNClient(object):
    def __init__(self, url, path, exclude_dirs=[".svn"], skipempty=True, verbose=True):
        self.client = pysvn.Client()
        self.url = url
        self.path = path
        self.exclude_dirs = exclude_dirs
        self.skipempty = skipempty
        self.verbose = verbose

    def reset(self, revision=None):
        self.revision = revision
        self.dirs = []
        self.paths = []

    def checkout(self, rev):
        log.info("checkout %s rev %s to %s " % (self.url, rev, self.path)) 
        self.client.checkout(self.url, self.path, revision=rev_(rev))

    def recurse(self, rev ):
        self.reset(rev)
        self.checkout(rev)
        leaves = crawler( self.path, self.path, self.dir_, self.exclude_dirs, self.skipempty )
        map(self.leaf_, leaves)
  
    def leaf_(self, rpath):
        self.paths.append(rpath)
        if self.verbose:
            print rpath

    def dir_(self, rpath):
        self.dirs.append(rpath)
        if self.verbose:
            print rpath + '/'

    def unprefixed_paths(self, prefix):
        return unprefix( self.paths , prefix )

    def unprefixed_dirs(self, prefix):
        return unprefix( self.dirs , prefix )

    def contents_digest(self):
        """
        #. called by compare_contents following a recurse 
           to update the working copy to a revision and crawl it to compile lists of dirs and paths
        #. `self.paths` are resolved relative to `self.hgdir`
        #.  special handling of symbolic links to mimic SVN 

        :return: dict keyed on path containing file content digests for all files resolved from `self.paths`  
        """
        def _digest(fp):
            md5 = hashlib.md5()
            for chunk in iter(lambda: fp.read(8192),''): 
                md5.update(chunk)
            return md5.hexdigest()
        pass
        def _resolve(p):
            if p[0] == '/':
                p = p[1:]
            path = os.path.join(self.path, p)
            log.debug("root %s p %s path %s " % (self.path, p, path ))
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


    def readlog(self, srctz=None, loctz=None):
        """
        """
        log.info("readlog")
        revlog = {}
        msg = self.client.log(self.path) 
        for m in msg:
            rev = m.data['revision'].number
            meta = dict([(_, m.data['revprops'].get("svn:%s"%_,None)) for _ in "log date author".split()])
            meta['srev'] = m.data['revision'].number
            dt = datetime.fromtimestamp(meta['date'], loctz)

            tloc = dt.replace(microsecond=0)
            tsrc = tloc.astimezone(srctz)

            meta['tsrc'] = tsrc
            meta['tloc'] = tloc
            meta['ssrc'] = int(meta['tsrc'].strftime("%s"))
            meta['sloc'] = int(meta['tloc'].strftime("%s"))

            revlog[rev] = meta
        pass
        self.log = revlog
        self.tlog = self.reindex( revlog, key='sloc')
        #IP.embed()

    def reindex(self, revlog, key='sloc'):
        """
        Convert revision keyed log records to be keyed by some other thing, 
        typically a timestamp

        :param revlog: dict of revision records keyed by revision number
        :param key: name of key to reindex on
        """
        klog = {}
        for srev,s in revlog.items():
            klog[s[key]] = s 
        pass
        #assert len(klog) == len(revlog) - 1 , "(-1 as rev 0 and 1 have same timestamp) need unique key for useful reindexing "
        return klog 



def parse(doc):
    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-v","--verbose", action="store_true")
    parser.add_argument("--skipempty", action="store_true")
    parser.add_argument("--loglevel",  default="INFO", help="")
    parser.add_argument("--url",  default="http://dayabay.phys.ntu.edu.tw/repos/env/trunk/", help="")
    parser.add_argument("--path", default="/tmp/subversion/env", help="")
    parser.add_argument("--srctz", default="utc", help="timezone of the SVN source timestamps, usually utc "  )
    parser.add_argument("--loctz", default="cst", help="timezone in which to make comparisons "  )
    parser.add_argument("-r","--revision", default="1", help="")  
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging,args.loglevel.upper()))
    return args 

def main():
    args = parse(__doc__)

    dtz = dict(cst=cst,utc=utc)
    srctz = dtz.get(args.srctz, None)
    loctz = dtz.get(args.loctz, None)

    sc = SVNClient(args.url, args.path, verbose=args.verbose, skipempty=args.skipempty)
    sc.readlog(srctz=srctz,loctz=loctz)

    rev = int(args.revision)
    sc.recurse(rev)


    IP.embed()
    


if __name__ == '__main__':
    main()


