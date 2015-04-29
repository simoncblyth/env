#!/usr/bin/env python
"""
SVNCLIENT.PY
==============

Control of SVN working copy via pysvn client.

#. trunk doesnt exist at revision 0
#. an alternative approach using direct access to SVN repo database 
   is in svncrawl.py 


Usage::

    (adm_env)delta:~ blyth$ svnclient.py -r724:730
    INFO:env.svn.bindings.svnclient:checkout http://dayabay.phys.ntu.edu.tw/repos/env/trunk/ rev 724 to /tmp/subversion/env 
    INFO:env.svn.bindings.svnclient:checkout http://dayabay.phys.ntu.edu.tw/repos/env/trunk/ rev 725 to /tmp/subversion/env 
    INFO:env.svn.bindings.svnclient:checkout http://dayabay.phys.ntu.edu.tw/repos/env/trunk/ rev 726 to /tmp/subversion/env 
    INFO:env.svn.bindings.svnclient:checkout http://dayabay.phys.ntu.edu.tw/repos/env/trunk/ rev 727 to /tmp/subversion/env 
    INFO:env.svn.bindings.svnclient:checkout http://dayabay.phys.ntu.edu.tw/repos/env/trunk/ rev 728 to /tmp/subversion/env 
    INFO:env.svn.bindings.svnclient:checkout http://dayabay.phys.ntu.edu.tw/repos/env/trunk/ rev 729 to /tmp/subversion/env 

Ignore Externals
------------------

env svn r731 (svn:externals testing) http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/731
adds svn:externals pointing at IHEP repos, causing credentials callback to be triggered.
Avoid this with::

    svnclient.py -r731 --ignore-externals

"""
import os, argparse, logging, hashlib, shutil
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
    def __init__(self, url, path, args ):
        """
        :param url: 
        :param path:

        :param clean_checkout_revs: empty string or comma delimited list of revisions 
        """
        assert len(path) > 5, "path [%s] sanity check fails" % path 
        if not path[-1] == '/':path = "%s/" % path   # ensure trailing slash, for root relative paths to avoid the leading slash

        self.url = url
        self.path = path

        self.exclude_dirs = [".svn"]
        self.skipempty = args.skipempty
        self.verbose = args.verbose
        self.ignore_externals = args.ignore_externals
        self.clean_checkout_revs = map(int, filter(None,args.clean_checkout_revs.split(",")))    
        self.known_bad_revs = map(int, filter(None,args.known_bad_revs.split(",")))    
        self.known_bad_paths = args.known_bad_paths.split(",")   


    def make_client(self):
        client = pysvn.Client()
        client.callback_get_login = self.get_login
        client.callback_conflict_resolver = self.conflict_resolver
        return client

    def get_login(self, realm, username, may_save ):
        """
        Avoid this partially implemented callback using::

           svnclient.py -r731 --ignore-externals

        """
        log.info("get_login realm %s username %s may_save %s " % (realm, username, may_save)) 
        retcode, username, password, save = "","","",""
        return retcode, username, password, save

    def conflict_resolver(self, conflict_description ):
        log.warn("conflict %s " % repr(conflict_description)) 
        #return conflict_choice, merge_file, save_merged
        return None


    def reset(self, revision=None):
        self.revision = revision
        self.dirs = []
        self.paths = []

    def is_knownbad(self):
        return self.revision in self.known_bad_revs
    def is_knownbadpath(self, path):
        return path in self.known_bad_paths

    def checkout(self, rev, clean=False):
        self.client = self.make_client()
        log.debug("checkout %s rev %s to %s clean %s " % (self.url, rev, self.path, clean)) 
        if clean:
            if os.path.exists(self.path):
                assert len(self.path) > 5, "sanity check %s " % self.path
                log.info("rmtree %s " % self.path )
                shutil.rmtree(self.path)
            pass
        self.client.checkout(self.url, self.path, revision=rev_(rev), ignore_externals=self.ignore_externals)
        status = self.client.status( self.path, get_all=False)
        for s in status:
            print s.data
        #IP.embed()



    def recurse(self, rev ):
        self.reset(rev)
        clean = rev in self.clean_checkout_revs
        self.checkout(rev, clean=clean)
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

    def contents_digest(self, filemap=None):
        """
        :param filemap: 

        When a Filemap instance is provides, the digest keys 
        are changed to renamed ones. 

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
            k = filemap.apply([p])[0] if not filemap is None else p
            if os.path.islink(path):
                digest[k] = mimic_svn_link_digest( path )
            else:
                with open(path,"rb") as fp:
                    digest[k] = _digest(fp)
            pass 
        return digest


    def readlog(self, srctz=None, loctz=None):
        """
        """
        log.info("readlog")
        revlog = {}
        self.client = self.make_client()
        msg = self.client.log(self.url) 
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
    remote = "http://dayabay.phys.ntu.edu.tw/repos/env/trunk/"
    local = "file:///var/scm/subversion/env/trunk/"

    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-v","--verbose", action="store_true")
    parser.add_argument("--skipempty", action="store_true")
    parser.add_argument("--ignore-externals", action="store_true")
    parser.add_argument("--loglevel",  default="INFO", help="")
    parser.add_argument("--url",  default=local, help="local file:// or remote http:// url of Subversion repository ")
    parser.add_argument("--path", default="/tmp/subversion/env", help="Directory of Subversion working copy")
    parser.add_argument("--srctz", default="utc", help="timezone of the SVN source timestamps, usually utc "  )
    parser.add_argument("--loctz", default="cst", help="timezone in which to make comparisons "  )
    parser.add_argument("--readlog", action="store_true", help=" "  )
    parser.add_argument("--clean-checkout-revs", default="", help="Comma delimited list of revisions for which clean checkouts are needed"  )
    parser.add_argument("--known-bad-revs", default="", help="Comma delimited list of revisions for which discrepancies are known"  )
    parser.add_argument("-i","--ipython", action="store_true", help=" "  )
    parser.add_argument("-r","--revision", default="1", help="")  
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging,args.loglevel.upper()))

    return args 


def revs_arg( arg ):
    if ":" in arg:
        bits = map(int,arg.split(":"))
        assert len(bits) == 2
        revs = range(*bits)
    else:
        revs = map(int,arg.split(","))
    pass
    return revs


def main():
    args = parse(__doc__)

    dtz = dict(cst=cst,utc=utc)
    srctz = dtz.get(args.srctz, None)
    loctz = dtz.get(args.loctz, None)

    sc = SVNClient(args.url, args.path, args )

    if args.readlog:
        sc.readlog(srctz=srctz,loctz=loctz)

    revs = revs_arg(args.revision)
    for rev in revs:
        sc.recurse(rev)

    if args.ipython:
        import IPython as IP
        IP.embed()
    


if __name__ == '__main__':
    main()


