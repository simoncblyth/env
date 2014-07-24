#!/usr/bin/env python
"""
::

   adm-
   compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 1002 --hgrev 1000 

   compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 10 --hgrev 8 

       # DUD svn revision 10, causes this to be the first to match 

::

    (adm_env)delta:~ blyth$ compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 1:10 --hgrev 0:9  
    INFO:env.scm.migration.compare_hg_svn:hgrev 0 svnrev 1 
    INFO:env.scm.migration.compare_hg_svn:hgrev 1 svnrev 2 
    INFO:env.scm.migration.compare_hg_svn:hgrev 2 svnrev 3 
    INFO:env.scm.migration.compare_hg_svn:hgrev 3 svnrev 4 
    INFO:env.scm.migration.compare_hg_svn:hgrev 4 svnrev 5 
    INFO:env.scm.migration.compare_hg_svn:hgrev 5 svnrev 6 
    INFO:env.scm.migration.compare_hg_svn:hgrev 6 svnrev 7 
    INFO:env.scm.migration.compare_hg_svn:hgrev 7 svnrev 8 
    INFO:env.scm.migration.compare_hg_svn:hgrev 8 svnrev 9       ## offset of 1 due to trunk restriction, up to dud svn rev 10

    compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 10 --hgrev 9 

    (adm_env)delta:~ blyth$ compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 10:19 --hgrev 8:17 
    INFO:env.scm.migration.compare_hg_svn:hgrev 8 svnrev 10 
    INFO:env.scm.migration.compare_hg_svn:hgrev 9 svnrev 11 
    INFO:env.scm.migration.compare_hg_svn:hgrev 10 svnrev 12 
    INFO:env.scm.migration.compare_hg_svn:hgrev 11 svnrev 13   
    INFO:env.scm.migration.compare_hg_svn:hgrev 12 svnrev 14 
    INFO:env.scm.migration.compare_hg_svn:hgrev 13 svnrev 15 
    INFO:env.scm.migration.compare_hg_svn:hgrev 14 svnrev 16 
    INFO:env.scm.migration.compare_hg_svn:hgrev 15 svnrev 17  
    INFO:env.scm.migration.compare_hg_svn:hgrev 16 svnrev 18       ## beyond the dud, need offset of 2 to match

    compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 10 --hgrev 8 -A 

    INFO:env.scm.migration.compare_hg_svn:hgrev 388 svnrev 390 
    lines_dirs
     [ r] /seed                
    lines_paths

    compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 10 --hgrev 8 -A  --skipempty 


    INFO:env.scm.migration.compare_hg_svn:hgrev 1598 svnrev 1600 
    ---------------------------------------------------------------------------
    HgException                               Traceback (most recent call last)
    ...
    HgException: Error running hg --cwd /tmp/mercurial/env update 1598:
    " + tErr: abort: case-folding collision between thho/NuWa/python/histogram/pyhist.py and thho/NuWa/python/histogram/PyHist.py


    compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 1600 --hgrev 1598 -A  --skipempty 

        ## keep getting this...


Argh case degenerate entries at SVN rev 1600::

    delta:~ blyth$ svncrawl.py /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --revision 1599 -v | grep -i PyHist
    /trunk/thho/NuWa/python/histogram/pyhist.py

    delta:~ blyth$ svncrawl.py /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --revision 1600 -v | grep -i PyHist
    /trunk/thho/NuWa/python/histogram/PyHist.py
    /trunk/thho/NuWa/python/histogram/pyhist.py

    delta:~ blyth$ svncrawl.py /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --revision 1601 -v | grep -i PyHist
    /trunk/thho/NuWa/python/histogram/PyHist.py

    delta:~ blyth$ svncrawl.py /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --revision 1602 -v | grep -i PyHist
    /trunk/thho/NuWa/python/histogram/PyHist.py

Manual fix::

    delta:~ blyth$ hg --cwd /tmp/mercurial/env update 1598
    abort: case-folding collision between thho/NuWa/python/histogram/pyhist.py and thho/NuWa/python/histogram/PyHist.py
    delta:~ blyth$ ll /tmp/mercurial/env/thho/NuWa/python/histogram/
    total 16
    drwxr-xr-x  10 blyth  wheel   340 Jul 24 20:17 ..
    -rw-r--r--   1 blyth  wheel  5258 Jul 24 20:17 pyhist.py
    drwxr-xr-x   3 blyth  wheel   102 Jul 24 20:21 .
    delta:~ blyth$ rm -rf /tmp/mercurial/env/thho/NuWa/python/histogram
    delta:~ blyth$ hg --cwd /tmp/mercurial/env update 1598
    1 files updated, 0 files merged, 0 files removed, 0 files unresolved
    delta:~ blyth$ ll /tmp/mercurial/env/thho/NuWa/python/histogram/
    total 16
    -rw-r--r--   1 blyth  wheel  5258 Jul 24 20:28 PyHist.py
    drwxr-xr-x  10 blyth  wheel   340 Jul 24 20:28 ..
    drwxr-xr-x   3 blyth  wheel   102 Jul 24 20:28 .
    delta:~ blyth$ 


SVN permits case degenerate paths to have distinct entries in its DB, but Mercurial doesnt.



"""
import os, logging, argparse
log = logging.getLogger(__name__)

from env.svn.bindings.svncrawl import SVNCrawler
from env.hg.bindings.hgcrawl import HGCrawler

def parse(doc):
    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-v","--verbose", action="store_true")
    parser.add_argument(     "--skipempty", action="store_true", help="skip empty SVN directories from the crawl " )
    parser.add_argument("-l","--loglevel", default="info")
    parser.add_argument("path", nargs=2 )
    parser.add_argument("--hgrev", default=None )
    parser.add_argument("--svnrev", default=None )
    parser.add_argument("--svnprefix", default="/trunk", help="path prefix to remove before comparison" )
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
    mismatch = filter(mismatch_, common_paths)

    check = {}
    check['hg_keys']  = sorted(hg_digest.keys()) == common_paths
    check['svn_keys'] = sorted(svn_digest.keys()) == common_paths
    check['zero mismatch'] = len(mismatch) == 0 
    check['digest match']  = svn_digest == hg_digest

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
    degenerates = {
        'pyhist':['/thho/NuWa/python/histogram/pyhist.py',
                  '/thho/NuWa/python/histogram/PyHist.py'],
    }

    svn_dirs = svn.unprefixed_dirs( svnprefix )
    common_dirs, hg_only_dirs, svn_only_dirs, lines_dirs  = compare_lists( hg.dirs, svn_dirs )

    svn_paths = svn.unprefixed_paths( svnprefix )
    common_paths, hg_only_paths, svn_only_paths, lines_paths = compare_lists( hg.paths, svn_paths )

    rl_svn_paths = list(reversed(map(lambda _:_.lower(), svn_paths )))  
    case_degenerate_ = lambda p:svn_paths.index(p) != len(svn_paths) - 1 - rl_svn_paths.index(p.lower())
    case_degenerates = filter( case_degenerate_ , svn_paths )

    check = {}
    check['case_degenerates'] = len(case_degenerates) == 0
    check['svn_only_paths'] = len(svn_only_paths) == 0
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
    hg  = HGCrawler(hgdir, verbose=args.verbose ) 
    svn = SVNCrawler(svndir, verbose=args.verbose, skipempty=args.skipempty) 

    if args.ALLREV:
        youngest_rev = svn.youngest_rev()
        svnrevs = range(int(args.svnrev),youngest_rev+1)
        hgrevs = range(int(args.hgrev),int(args.hgrev)+len(svnrevs))
    else:
        svnrevs = revs_arg(args.svnrev)
        hgrevs = revs_arg(args.hgrev)
    pass
    assert len(svnrevs) == len(hgrevs)  

    for hgrev, svnrev in zip(hgrevs,svnrevs):
        log.info("hgrev %s svnrev %s " % (hgrev, svnrev))
        hg.recurse(hgrev)   # updates hg working copy to this revision
        svn.recurse(svnrev)
        common_paths = compare_paths( hg, svn , args.svnprefix )
        compare_contents( hg , svn, args.svnprefix, common_paths )
    pass



if __name__ == '__main__':
    main()


