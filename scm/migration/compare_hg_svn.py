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
import IPython as IP
log = logging.getLogger(__name__)

from env.svn.bindings.svncrawl import SVNCrawler
from env.svn.bindings.svnclient import SVNClient
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

    parser.add_argument( "--svnpath", default="/tmp/subversion/env", help="")
    parser.add_argument( "--svnurl",  default="http://dayabay.phys.ntu.edu.tw/repos/env/trunk/", help="")

    parser.add_argument("--srctz", default="utc", help="timezone of the SVN source timestamps, usually utc "  )
    parser.add_argument("--loctz", default="cst", help="timezone in which to make comparisons "  )
    parser.add_argument("--svnprefix", default="", help="path prefix to remove before comparison, formerly /trunk but thats not needed with SVNClient" )
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
    svn_digest = svn.contents_digest()
    hg_digest = hg.contents_digest()
    mismatch_ = lambda _:hg_digest[_] != svn_digest[_]
    mismatch = filter(mismatch_, common_paths)     # matching only the common paths

    svn_only = list(set(svn_digest.keys()).difference(set(common_paths)))
    hg_only = list(set(hg_digest.keys()).difference(set(common_paths)))

    check = {}
    check['hg_keys']  = sorted(hg_digest.keys()) == common_paths
    check['svn_only'] = len(svn_only) == 0
    check['hg_only'] = len(hg_only) == 0
    check['common path mismatch'] = len(mismatch) == 0 

    issues = filter(lambda _:not _[1], check.items()) 

    if len(issues) > 0:
        log.info("issues encountered in compare_contents")
        import IPython
        IPython.embed()



def compare_paths( hg, svn, svnprefix, fmap=None, debug=False ):
    """
    :param hg: HGCrawler instance
    :param svn: SVNCrawler or SVNClient instance
    :param svnprefix: path prefix to be removed from SVN paths before comparison, eg /trunk when using SVNCrawler
    :param fmap: FileMap instance or None
 
    #. update hg working copy to the revision
    #. query SVN db for list of paths at the svnrev 

    #. Mercurial doesnt "do" empty directories 
    #. Comparison trips up on symbolic links, have handled symbolic links to files
       already but not to directories

    http://dayabay.phys.ntu.edu.tw/tracs/env/browser/trunk/qxml
    link db/bdbxml/qxml

    http://dayabay.phys.ntu.edu.tw/tracs/env/browser/trunk/db/bdbxml/qxml/


    """
    if svnprefix is "":
        svn_dirs = svn.dirs
        svn_paths = svn.paths
    else:
        svn_dirs = svn.unprefixed_dirs( svnprefix )
        svn_paths = svn.unprefixed_paths( svnprefix )
    pass

    common_dirs, hg_only_dirs, svn_only_dirs, lines_dirs  = compare_lists( hg.dirs, svn_dirs )

    if not fmap is None:
        svn_paths = fmap.apply( svn_paths )

    common_paths, hg_only_paths, svn_only_paths, lines_paths = compare_lists( hg.paths, svn_paths )

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
        pass

    #IP.embed()
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
        contiguous = range(min(svnrevs),max(svnrevs)+1)
        is_contiguous = contiguous == sorted(svnrevs)
        missing = list(set(contiguous).difference(set(svnrevs)))
        log.info("svnrevs min/max/count %s %s %s contiguous? %s missing %s " % (min(svnrevs),max(svnrevs),len(svnrevs), is_contiguous, repr(missing))) 

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
        #. updates hg working copy to this revision and crawls filesystem noting paths and dirs
        #. queries SVN database for this revision
          (hmm maybe should do based on working copy)
        """
        log.info("hgrev %s svnrev %s hgrev-svnrev %s " % (hgrev, svnrev, int(hgrev)-int(svnrev) ))
        self.hg.recurse(hgrev)  
        self.svn.recurse(svnrev)

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


class FileMap(object):
   def __init__(self, path ):
       chomp_ = lambda line:line.rstrip().lstrip()
       not_comment_ = lambda line:not line[0] == '#'
       with open(path,"r") as fp:
           lines = filter(not_comment_,map(chomp_,fp.readlines()))
       pass
       rename, include, exclude = self.parse_content(lines)
       self.rename = rename
       self.include = include
       self.exclude = exclude

   def apply(self, paths):
       pths = []
       for p in paths:
           if p in self.rename:
               pp = self.rename[p] 
           else:
               pp = p
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



def main():
    args = parse(__doc__)

    m_hgrev  = int(args.hgrev) if not args.hgrev is None else 0
    m_svnrev = int(args.svnrev) if not args.svnrev is None else 0

    hgdir = args.path[0]
    svndir = args.path[1]   # to backup SVN repo

    fmap = FileMap( args.filemap ) if not args.filemap is None else None
    hg  = HGCrawler(hgdir, verbose=args.verbose ) 

    #svn = SVNCrawler(svndir, verbose=args.verbose, skipempty=args.skipempty)     # direct access to backup SVN repo
    svn = SVNClient(args.svnurl, args.svnpath,  verbose=args.verbose, skipempty=args.skipempty)  # client working copy access

    cf = Compare( hg, svn, args )
    cf.readlog()

    for hgrev, svnrev in cf.revs:
        if hgrev < m_hgrev or svnrev < m_svnrev:continue 
        cf.recurse( hgrev, svnrev )

        #IP.embed()

        common_paths = compare_paths( hg, svn, args.svnprefix, fmap )
        compare_contents( hg , svn, args.svnprefix, common_paths )
    pass





if __name__ == '__main__':
    main()


