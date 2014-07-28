#!/usr/bin/env python
"""
::

   adm-
   compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 1002 --hgrev 1000 

   compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 10 --hgrev 8 

       # DUD svn revision 10, causes this to be the first to match 


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


def compare_log( hg, svn, hgrevs, svnrevs ):
    """
    """
    assert len(hgrevs) == len(svnrevs)
    youngest_rev = svn.youngest_rev()
    hgkeys = hg.log.keys() 
    svnkeys = svn.log.keys() 
    assert min(svnkeys) == 0 
    assert sorted(svnkeys) == range(0,max(svnkeys)+1), "expecting contiguous svnkeys revisions from 0 to maxrev "

    for hrev, srev in zip(hgrevs, svnrevs):
        h = hg.log[hrev]
        s = svn.log[srev]
        if str(h['log']) != str(s['log']):
            log.info("hrev %s srev %s log divergence hlog [%s] slog [%s] " % (hrev, srev, h['log'],s['log']) )


    import IPython
    IPython.embed()


def compare_paths( hg, svn, svnprefix, debug=False ):
    """
    :param hg: HGCrawler instance
    :param svn: SVNCrawler instance
    :param svnprefix: path prefix to be removed from SVN paths before comparison

    #. update hg working copy to the revision
    #. query SVN db for list of paths at the svnrev 

    Mercurial doesnt "do" empty directories 


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
            import IPython
            IPython.embed()
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

    dtz = dict(cst=cst,utc=utc)
    srctz = dtz.get(args.srctz, None)
    loctz = dtz.get(args.loctz, None)

    hg.readlog(srctz=srctz, loctz=loctz)
    svn.readlog(srctz=srctz, loctz=loctz)

    if args.ALLREV:
        youngest_rev = svn.youngest_rev()
        svnrevs = range(int(args.svnrev),youngest_rev+1)
        hgrevs = range(int(args.hgrev),int(args.hgrev)+len(svnrevs))
    else:
        svnrevs = revs_arg(args.svnrev)
        hgrevs = revs_arg(args.hgrev)
    pass
    assert len(svnrevs) == len(hgrevs)  

    compare_log( hg, svn, hgrevs, svnrevs )

    for hgrev, svnrev in zip(hgrevs,svnrevs):
        log.info("hgrev %s svnrev %s " % (hgrev, svnrev))
        hg.recurse(hgrev)   # updates hg working copy to this revision
        svn.recurse(svnrev)

        common_paths = compare_paths( hg, svn , args.svnprefix )
        compare_contents( hg , svn, args.svnprefix, common_paths )
    pass



if __name__ == '__main__':
    main()


