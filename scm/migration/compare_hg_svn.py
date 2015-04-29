#!/usr/bin/env python
"""
Full History Comparison of Subversion working copy with Mercurial 
===================================================================

Operates by:

#. matching Subversion and Mercurial revision logs, by commit times
#. traversing the revisions, checking them out and doing file digest comparisons
   between the Subversion and Mecurial versions

::

   compare_hg_svn.py hgdir svndir svnurl

   OR:

   adm-
   adm-compare-svnhg


TODO:

#. persist comparison status, perhaps in a pickled dict 
#. add status checks, to pysvn checkouts 
#. add hg verify calls

heprez comparison
-----------------

Compare paths tripping up on directories containing nothing but symbolic links::

    In [12]: svn_only_dirs
    Out[12]: ['bin', 'sources', 'sources/belle']

CONFIRMED to be due to the Subversion "feature" of not deleting non-empty folders 
which means that cannot go backwards in time with "svn checkout" alone in the
case of directories containing nothing but symbolic links.  

Workaround is to start the comparison from an empty SVN working copy folder and
check revisions monotonically.


tracdev comparison
-------------------

Expected effect of filemap repositionings::

    INFO:env.scm.migration.compare_hg_svn:1 ['svn_only_dirs'] issues encountered in compare_paths
    lines_dirs
     [ r] xsltmacro/branches   


"""
import os, logging, argparse
import IPython as IP
log = logging.getLogger(__name__)

from env.svn.bindings.svncrawl import SVNCrawler
from env.svn.bindings.svnclient import SVNClient
from env.hg.bindings.hgcrawl import HGCrawler
from env.scm.timezone import cst, utc

def compare_lists( l, r, verbose=False ):
    l = set(l)
    r = set(r)
    lines = []  
    for _ in l.union(r):  # either
        sl = "l" if _ in l else " " 
        sr = "r" if _ in r else " " 
        st = "%s%s" % ( sl, sr )
        if verbose or st != "lr":
            line = " [%s] %-20s " % ( st, _ )           
            lines.append(line)
        pass
    return sorted(list(l.intersection(r))),sorted(list(l.difference(r))),sorted(list(r.difference(l))), lines

class Compare(object):
    def __init__(self, hg, svn, args, filemap=None):
        """ 
        :param hg: HGCrawler instance
        :param svn: SVNClient instance
        :param args:
        """
        self.hg = hg
        self.svn = svn
        self.args = args
        self.filemap = filemap
        self.cf = {}

    def readlog(self):
        """
        Read the SVN and HG logs and establish revision 
        number mapping between them using the timestamp for identity matching 
        """
        dtz = dict(cst=cst,utc=utc)
        srctz = dtz.get(self.args.srctz, None)
        loctz = dtz.get(self.args.loctz, None)

        self.hg.readlog(srctz=srctz, loctz=loctz)
        self.svn.readlog(srctz=srctz, loctz=loctz)

        svnrevs = self.svn.log.keys() 
        contiguous = range(min(svnrevs),max(svnrevs)+1)
        is_contiguous = contiguous == sorted(svnrevs)
        missing = list(set(contiguous).difference(set(svnrevs)))
        log.info("readlog: svnrevs min/max/count %s %s %s contiguous? %s missing %s " % (min(svnrevs),max(svnrevs),len(svnrevs), is_contiguous, repr(missing))) 

        ho, so, co = self.compare_timestamps()
        self.dump_only( ho, so )
        
        revs, h2s, s2h = self.compare_log( co )

        self.revs = revs
        self.h2s = h2s
        self.s2h = s2h

        self.ho = ho
        self.so = so
        self.co = co

    def compare_timestamps(self):
        """
        Use timestamp keyed revison logs to find timestamps
        present in either/both hg and svn 

        ho, hg only
        so, svn only
        co, common
        """
        st = set(self.svn.tlog.keys())
        ht = set(self.hg.tlog.keys())
        ho = ht.difference(st)
        so = st.difference(ht)
        co = ht.intersection(st)
        #IP.embed()
        return ho, so, co

    def dump_only(self, ho, so ):
        """
        hg only entries explained by mismatches in repo snapshots 

        #. hg convert uses the live SVN repo, will be as uptodate as the convert
        #. SVN log access using the backup repo, will be as uptodate as the backup

        **So far** svn only entries are all explained by SVN 
        commits which do not constitute a hg revision, namely:

        #. create/delete empty folders only
        #. dud SVN revisions (r10 in env)
        #. change svn revision properties

        """
        log.info("hg only : %s entries" % len(ho))
        for t in sorted(ho):
            log.debug("%(hrev)s %(log)s " % self.hg.tlog[t])

        log.info("svn only : %s entries" % len(so))
        for t in sorted(so):
            log.debug("%(srev)s %(log)s " % self.svn.tlog[t])


    def compare_log(self, co):
        """
        :param co: list of timestamps in common between svn and hg

        Checks that the normalized log messages match and 
        creates mapping between 
        """
        def lognorm(l):
            lt = l.lstrip().rstrip()  # trim
            return " ".join(lt.split()) # collapse whitespace

        h2s = {}
        s2h = {}
        revs = []

        for t in sorted(co):
            hl = lognorm(self.hg.tlog[t]['log'])
            sl = lognorm(self.svn.tlog[t]['log'])
            hrev = self.hg.tlog[t]['hrev']
            srev = self.svn.tlog[t]['srev']
            revs.append( (hrev, srev,) )
            h2s[hrev] = srev
            s2h[srev] = hrev
            if hl != sl:
                print "hg/svn revision with common timestamp BUT with different log"
                print "h %-5s [%-70s]" % (hrev, hl) 
                print "s %-5s [%-70s]" % (srev, sl)
                assert 0, "after normalization the log entries should match "
            pass
        return revs, h2s, s2h

    def recurse(self, hgrev, svnrev ):
        """
        #. updates hg working copy to hgrev and crawls filesystem noting paths and dirs
        #. updated SVN working copy to svnrev and crawls filesystem noting paths and dirs
           (formerly used direct access to a backup SVN repo DB, this is faster but the
            comparison is easier and more like the usual usage pattern when using working copy)
        """
        log.info("hgrev %s svnrev %s hgrev-svnrev %s " % (hgrev, svnrev, int(hgrev)-int(svnrev) ))
        self.hg.recurse(hgrev)  
        self.svn.recurse(svnrev)

    def compare_paths( self, debug=False ):
        """
        #. update hg working copy to the revision

        #. Mercurial doesnt "do" empty directories 
        #. Comparison trips up on symbolic links, have handled symbolic links to files
           already but not to directories
        #. directories that contain only symbolic links not "seen" by Mercurial

        http://dayabay.phys.ntu.edu.tw/tracs/env/browser/trunk/qxml
        link db/bdbxml/qxml

        http://dayabay.phys.ntu.edu.tw/tracs/env/browser/trunk/db/bdbxml/qxml/
 
        """
        svnprefix = self.args.svnprefix
        if svnprefix is "":
            svn_dirs = self.svn.dirs
            svn_paths = self.svn.paths
        else:
            svn_dirs = self.svn.unprefixed_dirs( svnprefix )
            svn_paths = self.svn.unprefixed_paths( svnprefix )
        pass

        if not self.filemap is None:
            svn_dirs = self.filemap.apply( svn_dirs )
            svn_paths = self.filemap.apply( svn_paths )

        common_dirs,  hg_only_dirs,  svn_only_dirs_all ,  lines_dirs  = compare_lists( self.hg.dirs,  svn_dirs )
        common_paths, hg_only_paths, svn_only_paths, lines_paths = compare_lists( self.hg.paths, svn_paths )



        svn_only_dirs = filter( lambda _:not _ in self.args.expected_svnonly_dirs, svn_only_dirs_all )

        if not svn_only_dirs == svn_only_dirs_all:
            log.warn("noted expected svnonly dirs saved an error ")
            print "svn_only_dirs_all: %s " % repr(svn_only_dirs_all)
            print "svn_only_dirs    : %s " % repr(svn_only_dirs)
        pass

        check = {}
        check['hg_only_paths'] = len(hg_only_paths) == 0
        check['svn_only_dirs'] = len(svn_only_dirs) == 0
        check['hg_only_dirs'] = len(hg_only_dirs) == 0

        issues = dict(filter(lambda _:not _[1], check.items()))

        if len(issues) > 0:
            keys = issues.keys()
            log.info("%s %s issues encountered in compare_paths" % (len(keys),repr(keys)))
            print "lines_dirs\n", "\n".join(lines_dirs)  
            print "lines_paths\n", "\n".join(lines_paths)  
            if self.svn.is_knownbad():
                log.warn("known bad svn revision %s " % self.svn.revision )
            else:
                IP.embed()
            pass
        pass
        return common_paths

    def compare_contents( self, common_paths ):
        """
        Matching the content digests of common_paths

        :param: common_paths
        """
        svn_digest = self.svn.contents_digest(filemap=self.filemap)
        hg_digest = self.hg.contents_digest()

        mismatch_ = lambda _:hg_digest.get(_,None) != svn_digest.get(_,None)
        mismatch0 = filter(mismatch_, common_paths)     # matching only the common paths

        not_knownbadpath_ = lambda _:not self.svn.is_knownbadpath(_)
        mismatch = filter( not_knownbadpath_ , mismatch0 ) 

        kbp = list(set(mismatch0).difference(set(mismatch)))

        if len(kbp) > 0:
            log.debug("known bad paths excluded entries from mismatch %s " % repr(kbp))


        svn_only = list(set(svn_digest.keys()).difference(set(common_paths)))
        hg_only = list(set(hg_digest.keys()).difference(set(common_paths)))


        log.info("compare_contents paths %s svn_digest %s hg_digest %s mismatch %s  svn_only %s hg_only %s kbp %s " %
             (len(common_paths),len(svn_digest), len(hg_digest),len(mismatch), len(svn_only),len(hg_only), len(kbp)))

        check = {}
        check['hg_keys']  = sorted(hg_digest.keys()) == common_paths
        check['svn_only'] = len(svn_only) == 0
        check['hg_only'] = len(hg_only) == 0
        check['common path mismatch'] = len(mismatch) == 0 

        issues = filter(lambda _:not _[1], check.items()) 

        if len(issues) > 0:
            log.info("issues encountered in compare_contents : %s " % repr(issues))
            if self.svn.is_knownbad():
                log.warn("known bad svn revision %s " % self.svn.revision )
            else:
                IP.embed()
            pass

    def compare(self, hgrev, svnrev ):
        hs = (hgrev,svnrev)
        self.recurse(*hs)
        common_paths = self.compare_paths()
        self.compare_contents( common_paths )






class FileMap(object):
   def __init__(self, path ):
       chomp_ = lambda line:line.rstrip().lstrip()
       not_comment_ = lambda line:not line[0] == '#' 
       with open(path,"r") as fp:
           lines = filter(not_comment_,filter(None,map(chomp_,fp.readlines())))
       pass
       rename, include, exclude = self.parse_content(lines)
       self.rename = rename
       self.include = include
       self.exclude = exclude

   def apply(self, paths):
       """
       Mercurial migration tool `hg convert` has `--filemap` option supporting 
       path renaming.  During the creation of Mercurial repos from SVN repos for 
       example the paths are changed and thus old paths from SVN do not appear 
       in the HG repo.  

       Thus in order to verify the conversion by comparison of
       the source and destination paths and dirs it is necessary 
       to apply the filemap to the source paths in order to allow 
       comparison of the appropriate paths.  

       This method does that mapping.
       """
       pths = []
       for p in paths:
           pp = p
           for op,np in self.rename.items():
               if p.startswith(op):
                   pp = np + p[len(op):]
           pass
           if not pp == p:
               log.debug("filemap.apply changed [%s] => [%s] " % (p,pp))
           pass
           pths.append(pp)
       return pths

   def parse_content(self, lines):
       rename = {}
       include = []
       exclude = [] 

       for line in lines:
           elems = line.split()
           if elems[0] == 'rename' and len(elems)==3:
               rename[elems[1]] = elems[2]
           elif elems[0] == 'include' and len(elems) == 2:
               include.append(elems[1])
           elif elems[0] == 'exclude' and len(elems) == 2:
               exclude.append(elems[1])
           else:
               log.warn("ignoring unexpected line : %s " % line )
           pass
       return rename, include, exclude    






def parse(doc):
    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-l","--loglevel", default="info")
    parser.add_argument("-v","--verbose", action="store_true")
    parser.add_argument("path", nargs=3, help="Require 3 items: hgdir, svndir, svnurl "  )
    parser.add_argument("--hgrev", default=None, help="minimum hg revision to restrict comparisons whilst testing, does not need to map to --svnrev")
    parser.add_argument("--svnrev", default=None, help="minimum svn revision to restrict comparisons whilst testing, does not need to map to --hgrev" )
    parser.add_argument("--srctz", default="utc", help="timezone of the SVN source timestamps, usually utc "  )
    parser.add_argument("--loctz", default="cst", help="timezone in which to make comparisons "  )
    parser.add_argument("--svnprefix", default="", help="path prefix to remove before comparison, formerly /trunk but thats not needed with SVNClient" )
    parser.add_argument("--filemap", default=None, help="path to file containing include/exclude/rename directives" )
    parser.add_argument("--skipempty", action="store_true", help="skip empty SVN directories from the crawl " )
    parser.add_argument("--ignore-externals", action="store_true", help="ignore svn externals " )
    parser.add_argument("--expected-svnonly-dirs", default="", help="Comma delimited list of expected svnonly dirs to not raise error for. " )
    parser.add_argument("--clean-checkout-revs", default="", help="Comma delimited list of revisions for which clean checkouts are needed"  )
    parser.add_argument("--known-bad-revs", default="", help="Comma delimited list of revisions for which errors are ignored"  )
    parser.add_argument("--known-bad-paths", default="", help="Comma delimited list of paths for which errors are ignored"  )

    parser.add_argument("-A","--ALLREV", action="store_true", help="Switch on traversal of all revisions, this is slow.")
    args = parser.parse_args()
    args.m_hgrev  = int(args.hgrev) if not args.hgrev is None else 0
    args.m_svnrev = int(args.svnrev) if not args.svnrev is None else 0

    logging.basicConfig(level=getattr(logging,args.loglevel.upper()))
    return args 
 


def main():
    args = parse(__doc__)

    hgdir = args.path[0]
    svndir = args.path[1]   
    svnurl = args.path[2]   

    filemap = FileMap( args.filemap ) if not args.filemap is None else None

    hg  = HGCrawler( hgdir, verbose=args.verbose ) 
    svn = SVNClient( svnurl, svndir,  args )

    cf = Compare( hg, svn, args, filemap=filemap )
    cf.readlog()

    for hgrev, svnrev in cf.revs:
        if hgrev < args.m_hgrev or svnrev < args.m_svnrev:continue 
        cf.compare(hgrev, svnrev)       



if __name__ == '__main__':
    main()


