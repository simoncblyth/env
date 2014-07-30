#!/usr/bin/env python
"""
::

    hgcrawl.py /tmp/mercurial/env -r300

    hgcrawl.py /tmp/mercurial/env -r300 -m/
       #
       # dump the contents digest for all paths at the specified revision 


1603 hangs::

    (adm_env)delta:~ blyth$ hgcrawl.py /tmp/mercurial/env -r1603 -m/
    INFO:env.hg.bindings.hgcrawl:calling hg_update 1603 

1604 gives error::

    hgapi.hgapi.HgException: Error running hg --cwd /tmp/mercurial/env update 1604:
    " + tErr: abort: data/eve/ROOT/cmt/fragments/rootcint_dictionary.i@ef56e0aaa891: no match found!


Corrupted repo ?

* http://mercurial.selenic.com/wiki/RepositoryCorruption

Many errors, did the degeneracy handling deletions touch something it should not?::

    (adm_env)delta:env blyth$ hg verify
    checking changesets
    checking manifests
    crosschecking files in changesets and manifests
    checking files
     data/AbtViz/Aberdeen_World_extract.root.i@2878: missing revlog!
     data/AbtViz/Aberdeen_World_extract.root.d@2878: missing revlog!
     data/AbtViz/Abtfiles/Aberdeen_World_extract.root.i@3304: missing revlog!
     data/AbtViz/Abtfiles/ev.py.i@3304: missing revlog!

No verify errors in the bare repo into which the `hg convert` writes::

    (adm_env)delta:mercurial blyth$ hg-
    (adm_env)delta:mercurial blyth$ hg-convert
    hg convert --config convert.localtimezone=true --source-type svn --dest-type hg http://dayabay.phys.ntu.edu.tw/repos/env/trunk /var/scm/mercurial/env
    Mon Jul 28 13:46:57 CST 2014
    scanning source...
    sorting...
    converting...
    4 comparing env hg/svn history, find dud revision 10 
    3 machinery for new virtualenv adm- python, for sysadmin tasks like migarted to mercurial vs svn history comparisons 
    2 generalize tracmigrate into scmmigrate, investigate hgapi and svn bindings 
    1 svn and hg crawlers now check directory correspondence between revisions, not yet content 
    0 extend hg and svn crawlers to compare file content at all revisions, fix issues with symbolic links, problem of case degeneracy remains 
    Mon Jul 28 13:46:59 CST 2014

    (adm_env)delta:mercurial blyth$ cd /var/scm/mercurial/env   # bare repo, with just .hg dir
    (adm_env)delta:env blyth$ hg verify
    checking changesets
    checking manifests
    crosschecking files in changesets and manifests
    checking files
    4348 files, 4648 changesets, 13949 total revisions


Clone again to tmp::

    (adm_env)delta:mercurial blyth$ pwd
    /tmp/mercurial
    (adm_env)delta:mercurial blyth$ hg clone /var/scm/mercurial/env 
    destination directory: env
    updating to branch default
    3036 files updated, 0 files merged, 0 files removed, 0 files unresolved


* http://stackoverflow.com/questions/7595538/how-to-solve-a-mercurial-case-folding-collision


Perhaps cannot get away with deletions to avoid case collision.  
Maybe attempt to skip the problematic SVN revisions to avoid the problem instead.


"""
import os, argparse, logging, hashlib, re
log = logging.getLogger(__name__)
import hgapi

from datetime import datetime
from env.scm.timezone import cst 
from env.svn.bindings.svncommon import mimic_svn_link_digest


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
    #@classmethod 
    #def load_degenerates(cls, path):
    #    assert not path is None
    #    path = os.path.expanduser(path)
    #    chomp_ = lambda _:_[:-1]
    #    if os.path.exists(path):
    #       with open(path,"r") as fp:
    #          degenerate_paths = map(chomp_,fp.readlines())
    #    else:
    #        degenerate_paths = []  
    #    pass
    #    log.info("read degenerates from path %s " % path )
    #    return degenerate_paths

    def __init__(self, hgdir, verbose=False, exclude_dirs=[".hg"]):
        assert os.path.exists(os.path.join(hgdir,'.hg'))
        self.verbose = verbose
        self.hg = hgapi.Repo(hgdir)
        self.hgdir = hgdir
        self.exclude_dirs = exclude_dirs
        #self.degenerate_paths = degenerate_paths
        self.rootpath = hgdir
        self.hist = None

    def reset(self, revision=None):
        self.revision = revision
        self.dirs = []
        self.paths = []

    date_ptn = re.compile("([0123456789\.]*)([+-][\d]*)")

    def readlog(self, srctz=None, loctz=None, delim="@@@"):
        """
        :param srctz: source tzinfo instance, usually UTC
        :param loctz: local tzinfo instance
        :param delim: string delimiter, that must not be present in any log message, author name etc.. 

        Reads hg logs creating `self.log` revision keyed dict 
        and `self.tlog` timestamp keyed dict
        """
        log.info("readlog")
        revs = map(int,self.hg.hg_log(template="{rev}"+delim).split(delim)[:-1])
        desc = self.hg.hg_log(template="{desc}"+delim).split(delim)[:-1]
        auth = self.hg.hg_log(template="{author}"+delim).split(delim)[:-1]
        date = self.hg.hg_log(template="{date}"+delim).split(delim)[:-1]

        assert range(revs[0],-1,-1) == revs 
        assert len(revs) == len(desc) == len(auth) == len(date)

        revlog = {}
        for i,rev in enumerate(revs):
            dbit = self.date_ptn.match(date[i]).groups()    # 1178332617.0-28800 
            datz = map(float, dbit )            
            dat = datz[0] + datz[1]  #   add the tzoffset to give utc 

            tsrc = datetime.fromtimestamp(datz[0], srctz )
            tloc = tsrc.astimezone(loctz)

            log.debug("dbit %s datz %s  " % (repr(dbit), repr(datz)))      
            revlog[rev] = { 
                        'hrev':rev, 
                         'log':desc[i], 
                      'author':auth[i], 
                        'tloc':tloc, 
                        'tsrc':tsrc, 
                        'sloc':int(tloc.strftime("%s")), 
                        'ssrc':int(tsrc.strftime("%s")), 
                        }
            pass

        self.log = revlog
        self.tlog = self.reindex( revlog, key='sloc')

    def reindex(self, revlog, key='sloc'):
        """
        :param revlog: dict of revision records keyed by revision number
        :param key: name of key to reindex on
        """
        klog = {}
        for hrev,h in revlog.items():
            klog[h[key]] = h 
        pass
        assert len(klog) == len(revlog) , "need unique key for useful reindexing "
        return klog 

    def recurse(self, hgrev):
        """
        :param hgrev:

        Upadates Mercurial working copy to the revision specified
        and recursively crawls the filesystem
        """
        self.reset(hgrev)
        log.debug("calling hg_update %s " % hgrev )
        self.hg.hg_update(hgrev)
        crawler( self.hgdir, self.hgdir, self.dir_, self.leaf_, self.rootpath, self.exclude_dirs )

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
    parser.add_argument( "--degenerates", default=None,  )
    parser.add_argument( "-m","--md5", default=None )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging,args.loglevel.upper()))
    return args 
    
def main():
    args = parse(__doc__)
 

    hc = HGCrawler(args.path[0], verbose=args.verbose, exclude_dirs=[".hg"])
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


