#!/usr/bin/env python
"""
::

   compare_hg_svn.py /tmp/mercurial/env /var/scm/backup/cms02/repos/env/2014/07/20/173006/env-4637 --svnrev 1002 --hgrev 1000 




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
    return lines



def compare( hg, svn ):
    """
    #. update hg working copy to the revision
    #. query SVN db for list of paths at the svnrev 

    Mercurial doesnt "do" empty directories 
    """
    print hg
    print svn 

    ddirs  = compare_lists( hg.dirs, svn.dirs )
    print "ddirs\n", "\n".join(ddirs)  

    dpaths = compare_lists( hg.paths, svn.paths )
    print "dpaths\n", "\n".join(dpaths)  

    import IPython
    IPython.embed()



def main():
    args = parse(__doc__)

    hgdir = args.path[0]
    svndir = args.path[1]

    assert os.path.exists(os.path.join(hgdir,'.hg'))
    assert os.path.exists(os.path.join(svndir,'db/revs'))

    hg  = HGCrawler(hgdir, verbose=args.verbose ) 
    svn = SVNCrawler(svndir, verbose=args.verbose,skipempty=args.skipempty) 

    hg(args.hgrev)   # updates hg working copy to this revision
    svn(args.svnrev)

    compare( hg, svn )





if __name__ == '__main__':
    main()


