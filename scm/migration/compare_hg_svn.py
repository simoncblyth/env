#!/usr/bin/env python
"""
::

   adm-
   compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 1002 --hgrev 1000 

   compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 10 --hgrev 8 

       # DUD svn revision 10, causes this to be the first to match 


Mysteries:

#. initial trunk only convert, had almost all revisions converted with clearer match
   (simple constant revision offset) after the dud ? 
   doing full convert has many dropouts for empties ? making the relationship
   between SVN and HG revisions to require mappings.



"""
import os, logging, argparse
log = logging.getLogger(__name__)

from env.svn.bindings.svncrawl import SVNCrawler
from env.hg.bindings.hgcrawl import HGCrawler
from env.scm.timezone import cst, utc

def parse(doc):
    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-v","--verbose", action="store_true")
    parser.add_argument(     "--skipempty", action="store_true", help="skip empty SVN directories from the crawl " )
    parser.add_argument("-l","--loglevel", default="info")
    parser.add_argument("path", nargs=2 )
    parser.add_argument("--hgrev", default=None )
    parser.add_argument("--svnrev", default=None )
    parser.add_argument("--srctz", default="utc", help="timezone of the SVN source timestamps, usually utc "  )
    parser.add_argument("--loctz", default="cst", help="timezone in which to make comparisons "  )
    parser.add_argument("--svnprefix", default="/trunk", help="path prefix to remove before comparison" )
    parser.add_argument("--degenerates", default=None, help="path to file containg list of degenerate paths to be unlinked prior to each hg update" )
    parser.add_argument("--filemap", default=None, help="path to file containing include/exclude/rename directives" )
    parser.add_argument("-A","--ALLREV", action="store_true", help="Switch on traversal of all revisions, this is slow.")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging,args.loglevel.upper()))
    return args 
 

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

def compare_contents( hg, svn , svnprefix, common_paths ):
    pass
    svn_digest = svn.read_contents_digest()
    hg_digest = hg.contents_digest()
    mismatch_ = lambda _:hg_digest[_] != svn_digest[_]
    mismatch = filter(mismatch_, common_paths)     # matching only the common paths

    svn_only = list(set(svn_digest.keys()).difference(set(common_paths)))
    hg_only = list(set(hg_digest.keys()).difference(set(common_paths)))

    is_degenerate_ = lambda p:p in hg.degenerate_paths
    degenerate_svn_only = filter( is_degenerate_, svn_only )

    check = {}
    check['hg_keys']  = sorted(hg_digest.keys()) == common_paths
    check['svn extras all degenerates'] = len(degenerate_svn_only) == len(svn_only)  
    check['common path mismatch'] = len(mismatch) == 0 

    issues = filter(lambda _:not _[1], check.items()) 

    if len(issues) > 0:
        log.info("issues encountered in compare_contents")
        import IPython
        IPython.embed()



def compare_paths( hg, svn, svnprefix, debug=False ):
    """
    :param hg: HGCrawler instance
    :param svn: SVNCrawler instance
    :param svnprefix: path prefix to be removed from SVN paths before comparison

    #. update hg working copy to the revision
    #. query SVN db for list of paths at the svnrev 

    #. Mercurial doesnt "do" empty directories 
    #. Comparison trips up on symbolic links, have handled symbolic links to files
       already but not to directories

    http://dayabay.phys.ntu.edu.tw/tracs/env/browser/trunk/qxml
    link db/bdbxml/qxml

    http://dayabay.phys.ntu.edu.tw/tracs/env/browser/trunk/db/bdbxml/qxml/

    Even with skipempty, this is still tripping up:: 

        INFO:env.scm.migration.compare_hg_svn:hgrev 644 svnrev 646 
        INFO:env.svn.bindings.svncrawl:skipempty dir /trunk/thho/root 
        INFO:env.svn.bindings.svncrawl:skipempty dir /trunk/dyb/gaudi 
        INFO:env.svn.bindings.svncrawl:skipempty dir /trunk/dyb/external 
        lines_dirs
         [ r] /thho                
        lines_paths

        INFO:env.scm.migration.compare_hg_svn:hgrev 1444 svnrev 1446 
        INFO:env.svn.bindings.svncrawl:skipempty dir /trunk/unittest/nose/html 
        INFO:env.svn.bindings.svncrawl:skipempty dir /trunk/macros/aberdeen 
        INFO:env.svn.bindings.svncrawl:skipempty dir /trunk/thho/NuWa/python 
        lines_dirs
         [ r] /thho/NuWa           
        lines_paths

        INFO:env.scm.migration.compare_hg_svn:issues encountered in compare_paths

    """
    svn_dirs = svn.unprefixed_dirs( svnprefix )
    common_dirs, hg_only_dirs, svn_only_dirs, lines_dirs  = compare_lists( hg.dirs, svn_dirs )

    svn_paths = svn.unprefixed_paths( svnprefix )
    common_paths, hg_only_paths, svn_only_paths, lines_paths = compare_lists( hg.paths, svn_paths )

    rl_svn_paths = list(reversed(map(lambda _:_.lower(), svn_paths )))  
    case_degenerate_ = lambda p:svn_paths.index(p) != len(svn_paths) - 1 - rl_svn_paths.index(p.lower())
    case_degenerates = filter( case_degenerate_ , svn_paths )

    is_known_degenerate_ = lambda p:p in hg.degenerate_paths
    degenerate_svn_only_paths = filter( is_known_degenerate_, svn_only_paths )

    check = {}
    check['svn_only_paths'] = len(svn_only_paths) == len(degenerate_svn_only_paths)
    check['hg_only_paths'] = len(hg_only_paths) == 0
    check['svn_only_dirs'] = len(svn_only_dirs) == 0
    check['hg_only_dirs'] = len(hg_only_dirs) == 0

    issues = dict(filter(lambda _:not _[1], check.items()))

    if len(issues) > 0:
        keys = issues.keys()
        if len(issues) == 2 and 'case_degenerates' in keys and 'svn_only_paths' in keys and  svn_only_paths == case_degenerates:
            log.info("compare_paths : known problem of case_degenerates %s " % repr(case_degenerates) ) 
        else:
            log.info("%s %s issues encountered in compare_paths" % (len(keys),repr(keys)))
            print "lines_dirs\n", "\n".join(lines_dirs)  
            print "lines_paths\n", "\n".join(lines_paths)  
            #import IPython
            #IPython.embed()
        pass
    return common_paths



def revs_arg( arg ):
    if ":" in arg:
        bits = map(int,arg.split(":"))
        assert len(bits) == 2
        revs = range(*bits)
    else:
        revs = map(int,arg.split(","))
    pass
    return revs


class Compare(object):
    def __init__(self, hg, svn, args ):
        """ 
        :param hg: HGCrawler instance
        :param svn: SVNCrawler instance
        :param args:
        """
        self.hg = hg
        self.svn = svn
        self.args = args

    def readlog(self):
        """
        #. read the SVN and HG logs and establish revision mapping 
           using the timestamp  
        """
        dtz = dict(cst=cst,utc=utc)
        srctz = dtz.get(self.args.srctz, None)
        loctz = dtz.get(self.args.loctz, None)

        self.hg.readlog(srctz=srctz, loctz=loctz)
        self.svn.readlog(srctz=srctz, loctz=loctz)

        svnrevs = self.svn.log.keys() 
        assert min(svnrevs) == 0 
        assert sorted(svnrevs) == range(0,max(svnrevs)+1), "expecting contiguous svn revisions from 0 to maxrev "

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
        st = set(self.svn.tlog.keys())
        ht = set(self.hg.tlog.keys())
        ho = ht.difference(st)
        so = st.difference(ht)
        co = ht.intersection(st)
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


    def revisions(self):  
        if self.args.ALLREV:
            youngest_rev = self.svn.youngest_rev()
            svnrevs = range(int(self.args.svnrev),youngest_rev+1)
            hgrevs = range(int(self.args.hgrev),int(self.args.hgrev)+len(svnrevs))
        else:
            svnrevs = revs_arg(self.args.svnrev)
            hgrevs = revs_arg(self.args.hgrev)
        pass
        assert len(svnrevs) == len(hgrevs)  
        return zip(hgrevs, svnrevs)







def main():
    args = parse(__doc__)
    hgdir = args.path[0]
    svndir = args.path[1]

    if not args.degenerates is None:
        degenerate_paths = HGCrawler.load_degenerates( args.degenerates )
    else:
        degenerate_paths = []
    pass

    hg  = HGCrawler(hgdir, verbose=args.verbose, degenerate_paths=degenerate_paths ) 
    svn = SVNCrawler(svndir, verbose=args.verbose, skipempty=args.skipempty) 

    cf = Compare( hg, svn, args )
    cf.readlog()

    for hgrev, svnrev in cf.revs:
        log.info("hgrev %s svnrev %s " % (hgrev, svnrev))
        hg.recurse(hgrev)   # updates hg working copy to this revision
        svn.recurse(svnrev)

        common_paths = compare_paths( hg, svn , args.svnprefix )
        compare_contents( hg , svn, args.svnprefix, common_paths )
    pass



if __name__ == '__main__':
    main()


