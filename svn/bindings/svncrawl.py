#!/usr/bin/python
"""
Usage::

    svncrawl.py -v -r1000 /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637
        #
        # recursively crawl the repository, path names at the specified revision
        # NB verbose option -v needed to list all the paths
        #

    svncrawl.py -r1000 /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 -c /trunk/root/formula1r.C
    svncrawl.py -r2000 /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 -c /trunk/env.bash
        #
        # cat the specified path to stdout, corresponding to the version 
        # at the specified revision  
        # 
        # NB full path including the /trunk prefix is required
        #

    svncrawl.py /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 -m/ --ALLREV --pickle
        # 
        # calculate digests for all files at all revisions from 0 to youngest, 
        # writing pickle dicts keyed on filepath (including the /trunk)
        #
        # Writing env pickles from 0.pk to 4637.pk took 10 mins on D 
        #

    svncrawl.py /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 -K -r3001 -m/trunk/rootmq/include/rootmq_LinkDef.hh
    197d7199244706b55e63179bf08b5ba3

        # -K/--readpickle mode reads the cached pickle files rather than the SVN repository DB 



Adopted monolthic style to avoid SWIG segfaulting, this avoids wrapping head around
SWIG python memory management again (I did this previously, possibly with BerkelyDB xmldb 
interface investigations for heprez).


Symbolic Links
---------------

http://en.wikipedia.org/wiki/Apache_Subversion

*svn:special*
 
    This property is not meant to be set or modified directly by users. As of 2010
    it is only used for having symbolic links in the repository. When a symbolic
    link is added to the repository, a file containing the link target is created
    with this property set. When a Unix-like system checks out this file, the
    client converts it to a symbolic link.

SVN stores links as files starting link and containing the target path::

    In [26]: hashlib.md5("link module.py").hexdigest() == svn_digest['/unittest/demo/package/module_test.py']
    Out[26]: True

    In [29]: hashlib.md5("link ../xmlplug.py").hexdigest() == svn_digest['/unittest/nose/xml_plugin/xmlplug.py']
    Out[29]: True


Log Comparison
------------------

::

    delta:e blyth$ svn log -r2 -v
    ------------------------------------------------------------------------
    r2 | blyth | 2007-05-05 10:36:57 +0800 (Sat, 05 May 2007) | 1 line
    Changed paths:


Based on examples from 

* http://svnbook.red-bean.com/en/1.7/svn.developer.usingapi.html
* http://jtauber.com/python_subversion_binding/

"""
import sys, argparse, logging, hashlib, pickle, re
from datetime import datetime
import os.path
import svn.fs, svn.core, svn.repos

from svncommon import unprefix

log = logging.getLogger(__name__)

def crawl_filesystem_dir(root, directory, dir_, leaf_, exclude_dirs=['tags','branches'], rootpath='/trunk', skipempty=False):
    """
    Recursive crawler of SVN repository 

    #. cannot just check for no entries to detect empties, 
       as folders containing nothing but other empty folders would be 
       considered non-empty unlike Mercurial behavior 

    """
    entries = svn.fs.svn_fs_dir_entries(root, directory)
    names = entries.keys()
    reldir = directory[len(rootpath):]  

    leaves = []
    for name in names:
        full_path = directory + '/' + name
        if svn.fs.svn_fs_is_dir(root, full_path):
            if not name in exclude_dirs:
                subleaves = crawl_filesystem_dir(root, full_path, dir_, leaf_, exclude_dirs, rootpath, skipempty)
                leaves.extend(subleaves)
        else:
            leaves.append(full_path[len(rootpath):])
        pass
    pass

    if len(leaves) == 0 and skipempty:
        log.debug("skipempty dir %s " % reldir )
    else:
        log.debug("dir %s has %s leaves " % (reldir,len(leaves)) )
        dir_(reldir)
    pass
    return leaves 




class SVNCrawler(object):
    def __init__(self, repos_path, exclude_dirs=['tags','branches'], rootpath="", verbose=False, skipempty=False):
        """
        :param repos_path:
        :param exclude_dirs:
        :param rootpath:
        """
        assert os.path.exists(os.path.join(repos_path,'db/revs'))

        self.repos_path = repos_path
        self.exclude_dirs = exclude_dirs
        self.rootpath = rootpath
        self.verbose = verbose
        self.skipempty = skipempty
        self.log = None
        self.reset()

    def contents_digest( self, revision=None, prefix='/trunk' ): 
        """
        :param revision: defaults to revision of the prior recurse
        :return: dict keyed by path with content digests for each file
        """ 
        if revision is None:
            revision = self.revision
        pass
        assert not revision is None
        digest = pickle_load( self.repos_path, revision ) 
        if not prefix is None:
            return dict(zip(unprefix(digest.keys(),prefix),digest.values())) 

        return digest

    def reset(self, revision=None):
        self.revision = revision
        self.paths = []
        self.dirs = []

    def unprefixed_paths(self, prefix):
        return unprefix( self.paths , prefix )

    def unprefixed_dirs(self, prefix):
        return unprefix( self.dirs , prefix )

    def youngest_rev(self): 
        repos_obj = svn.repos.svn_repos_open(self.repos_path)
        fs_obj = svn.repos.svn_repos_fs(repos_obj)
        rev = svn.fs.svn_fs_youngest_rev(fs_obj) 
        return rev


    date_ptn = re.compile("(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})T(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})\.(?P<subsec>\d*)Z")

    def readlog(self, srctz=None, loctz=None):
        """
        :param loctz: tzinfo instance in which to make timestamp comparisons 
        :param srctz: tzinfo instance for source SVN timestamps 
        """
        repos_obj = svn.repos.svn_repos_open(self.repos_path)
        fs_obj = svn.repos.svn_repos_fs(repos_obj)
        maxrev = svn.fs.svn_fs_youngest_rev(fs_obj) 
        revlog = {}
        log.info("readlog srctz %s loctz %s  to maxrev %s " % (repr(srctz),repr(loctz),maxrev) ) 
        for rev in range(0,maxrev+1):
            meta = {}
            for propname in "log date author".split(): 
                propval = svn.fs.revision_prop(fs_obj, rev, "svn:%s"%propname)
                meta[propname] = propval
            pass
            d = self.date_ptn.match(meta['date']).groupdict()
            tk = "year month day hour minute second".split()
            tv = map(lambda k:int(d[k]), tk)
            dt = datetime(*tv)
            meta['srev'] = rev
            meta['tsrc'] = dt.replace(tzinfo=srctz)
            meta['tloc'] = meta['tsrc'].astimezone(loctz)
            meta['ssrc'] = meta['tsrc'].strftime("%s")
            meta['sloc'] = meta['tloc'].strftime("%s")
            revlog[rev] = meta
        pass
        self.log = revlog
        self.tlog = self.reindex( revlog, key='sloc')

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
        assert len(klog) == len(revlog) - 1 , "(-1 as rev 0 and 1 have same timestamp) need unique key for useful reindexing "
        return klog 

    def recurse(self, revision=None):
        """
        Passing root_obj between scopes causes segmentation faults, 
        SWIG can be that way.
        """
        repos_obj = svn.repos.svn_repos_open(self.repos_path)
        fs_obj = svn.repos.svn_repos_fs(repos_obj)
        rev = svn.fs.svn_fs_youngest_rev(fs_obj) if revision is None else int(revision)
        root_obj = svn.fs.svn_fs_revision_root(fs_obj, rev)

        self.reset(revision)
        leaves = crawl_filesystem_dir(root_obj, self.rootpath, self.dir_, self.leaf_, self.exclude_dirs, self.rootpath, self.skipempty)
        map(self.leaf_, leaves)
        
 
    def leaf_(self, rpath):
        self.paths.append(rpath)
        if self.verbose:
            print rpath

    def dir_(self, rpath):
        self.dirs.append(rpath)
        if self.verbose:
            print rpath + '/'

    def file_contents(self, path, revision=None, cat=False, dig=False):
        """
        Ugly monolithic style as SWIG is prone to Segfaulting 
        at scope boundaries.
        """
        repos_obj = svn.repos.svn_repos_open(self.repos_path)
        fs_obj = svn.repos.svn_repos_fs(repos_obj)
        rev = svn.fs.svn_fs_youngest_rev(fs_obj) if revision is None else int(revision)
        log.info("file_contents revision %s rev %s path [%s] " % (revision,rev,path))
        root_obj = svn.fs.svn_fs_revision_root(fs_obj, rev)

        digest = {}

        def _digest(fp):
            md5 = hashlib.md5()
            for chunk in iter(lambda: fp.read(8192),''):
                md5.update(chunk)
            return md5.hexdigest()

        def _write(fp):
            for chunk in iter(lambda: fp.read(8192),''):
                sys.stdout.write(chunk)

        if path is '/': 
            self.recurse(revision)
            paths = self.paths
        else:
            paths = [path]
        pass


        for p in paths:
            props = svn.fs.node_proplist(root_obj, p)
            special = svn.fs.node_prop(root_obj, p, 'svn:special')
            if special == '*' or len(props) > 0:
                log.info("p %s props %s special %s " % (p,repr(props),repr(special)))

            s = svn.fs.file_contents(root_obj, p)
            fp = svn.core.Stream(s)
            if cat:
                _write(fp)
            elif dig:
                digest[p] = _digest(fp)
            else:
                pass 

        return digest

    def cat(self, path, revision=None):
        self.file_contents( path, revision, cat=True, dig=False)

    def md5(self, path, revision=None):
        digest = self.file_contents( path, revision, cat=False, dig=True)
        return digest 


    def __repr__(self):
        return "%s %s %s dirs %s leaves %s " % (self.__class__.__name__, self.repos_path, self.revision,len(self.dirs),len(self.paths))

def parse(doc):
    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-v","--verbose", action="store_true")
    parser.add_argument(     "--skipempty", action="store_true")
    parser.add_argument(     "--loglevel", default="info")
    parser.add_argument(     "--ALLREV", action="store_true", help="Switch on traversal of all revisions, this is slow.")
    parser.add_argument( "-K","--readpickle", action="store_true", help="Read from pickle files, not from SVN repo database")
    parser.add_argument(      "--readlog", action="store_true", help="Read svn log info")
    parser.add_argument(     "--pickle", action="store_true", help="Switch on writing of digest pickle file")
    parser.add_argument(     "--pickle-path", default=None, help="Defaults to position one up from the repos_path")
    parser.add_argument( "-c", "--cat", default=None, help="path of node contents at the specified revision to cat to stdout ") 
    parser.add_argument( "-m", "--md5", default=None, help="path of node contents at the specified revision to compute digest for, or / to denote all ") 
    parser.add_argument( "path", nargs=1 )
    parser.add_argument( "-r","--revision", default=None )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging,args.loglevel.upper()))
    return args 
   

 
def pickle_path( repos_path, revision ):
    pkdir = os.path.dirname(repos_path)
    pknam = os.path.basename(repos_path)
    return os.path.join( pkdir, "%s.digest" % pknam, "%s.pk" % revision )

def pickle_write( digest, repos_path, revision  ):
    pkp = pickle_path( repos_path, revision )
    log.info("writing digest to %s " % pkp ) 
    pkd = os.path.dirname(pkp)
    if not os.path.exists(pkd):
        os.makedirs(pkd) 
    pass
    pickle.dump( digest , file(pkp, "wb") , pickle.HIGHEST_PROTOCOL )


def pickle_load( repos_path, revision ):
    pkp = pickle_path( repos_path, revision )
    digest = pickle.load( file(pkp,"rb") ) 
    return digest  


def main():
    args = parse(__doc__)
    repos_path = svn.core.svn_dirent_canonicalize(args.path[0])
    sc = SVNCrawler(repos_path, verbose=args.verbose, skipempty=args.skipempty)
    if args.readpickle:
        digest = sc.contents_digest( args.revision ) 
        if path in digest:
            print digest[path]
        else:
            print digest 
        return

    youngest_rev = sc.youngest_rev()

    log.info("created SVNCrawler repos_path %s youngest_rev %s " % (repos_path, youngest_rev) ) 
    if not args.cat is None:
        sc.cat(args.cat, args.revision)  # print emits trailing newline
    elif not args.md5 is None:
        if args.ALLREV:
            revisions = range(0,youngest_rev+1) 
        else:
            revisions = [args.revision]
        pass
        for revision in revisions: 
            digest = sc.md5(args.md5, revision)
            if args.verbose:
                print digest 
            if args.pickle:
                pickle_write( digest, repos_path, revision )
            pass
        pass 
    elif args.readlog:
        sc.readlog()
        import IPython
        IPython.embed()
    else:
        sc.recurse(args.revision)
        print sc
    pass
    log.debug("exit") 


if __name__ == "__main__":
    main()




