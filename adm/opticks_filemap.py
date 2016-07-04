#!/usr/bin/env python
"""
Opticks Filemap Creator
=========================

* https://www.mercurial-scm.org/wiki/MercurialApi

When using *hg convert* to spawn the opticks repo
from the overlarge env repo find that the convert explicitly 
includes only history within the list of folders provided.

Thus construct a useful history in the spawned repositor, 
need to obtain a list of dirs that files of interest have resided in 
since their birth.  

The set of current opticks files is an appropriate starting point.
This script automate the construction of such a list of folders
using the dis-encouraged internal Mercurial API. 

Preparation
------------

Make a separate envtmp clone for safety::

    delta:~ blyth$ hg clone http://bitbucket.org/simoncblyth/env envtmp
    real URL is https://bitbucket.org/simoncblyth/env
    requesting all changes
    adding changesets
    adding manifests
    adding file changes
    added 6131 changesets with 31753 changes to 8640 files
    updating to branch default
    5035 files updated, 0 files merged, 0 files removed, 0 files unresolved


Remember to pull/update envtmp following env changes
-----------------------------------------------------

::

    (adm_env)delta:env blyth$ cd ~/envtmp/
    (adm_env)delta:envtmp blyth$ pwd
    /Users/blyth/envtmp
    (adm_env)delta:envtmp blyth$ hg pull
    real URL is https://bitbucket.org/simoncblyth/env
    pulling from http://bitbucket.org/simoncblyth/env
    searching for changes
    adding changesets
    adding manifests
    adding file changes
    added 8 changesets with 26 changes to 23 files
    (run 'hg update' to get a working copy)
    (adm_env)delta:envtmp blyth$ hg up
    23 files updated, 0 files merged, 12 files removed, 0 files unresolved



When did Opticks development begin ? 4910 [2015-01-20] "try out NVIDIA Optix 301"
-----------------------------------------------------------------------------------

* https://bitbucket.org/simoncblyth/env/commits/6130  externals dir collecting opticks externals, populated by opticks-xcollect
* https://bitbucket.org/simoncblyth/env/commits/5130 [2015-05-16] Cerenkov photon generation arrives into OptiX context, photon distributions look familiar at a glance
* https://bitbucket.org/simoncblyth/env/commits/4130 [2013-11-26] investigating how to implement node navigation, view bookmarking etc.. inside meshlab

Tail of env.bash is good way to see whats happening.

* https://bitbucket.org/simoncblyth/env/src/4910/env.bash   OptiX precursor appears

* https://bitbucket.org/simoncblyth/env/commits/4910/   2015-01-20  try out NVIDIA Optix 301
* https://bitbucket.org/simoncblyth/env/commits/7a51deb30659403a1a616bbb27da67dccd3b92c8



Discovering the API
----------------------

::

    In [24]: len(list(repo.changelog.revs()))
    Out[24]: 211

    In [28]: repo[0]
    Out[28]: <changectx d34b880a5b5a>

    In [29]: repo[210]
    Out[29]: <changectx 3e12299774c7>

    In [30]: repo[211]
    Out[30]: <changectx 000000000000>

    In [31]: repo[212]
    RepoLookupError: unknown revision '212'

Hmm not a changeset its all files in repo at that revision::

    In [35]: print "\n".join([f for f in repo[210]])
    CMakeLists.txt
    Makefile
    assimprap/ASIRAP_API_EXPORT.hh
    assimprap/ASIRAP_BODY.hh
    assimprap/ASIRAP_HEAD.hh
    assimprap/ASIRAP_LOG.cc
    ...

Use files method to see changeset::

    In [47]: repo[100].files()
    Out[47]: ['optickscore/OpticksEvent.cc', 'optickscore/OpticksEvent.hh']

    In [48]: repo[100].description()
    Out[48]: 'prepare for migrating NumpyEvt up to OpticksEvent'

    In [78]: repo[210]['opticks.bash']
    Out[78]: <filectx opticks.bash@3e12299774c7>

    In [79]: fc = repo[210]['opticks.bash']

    In [84]: fc.parents()
    Out[84]: [<filectx opticks.bash@4a3b091c9adb>]

    In [85]: fc.parents()[0]
    Out[85]: <filectx opticks.bash@4a3b091c9adb>

    In [128]: repo[lrev-10]["opticksnpy/NPY.hpp"]
    Out[128]: <filectx opticksnpy/NPY.hpp@6611e08d62cc>


Revsets

* https://selenic.com/hg/help/revsets


"""
import os, argparse, logging
log = logging.getLogger(__name__)
from mercurial import ui, hg  ## dis-encouraged API

class Follower(object):
    def __init__(self, repo, exclude=[], firstrev=0):
        self.repo = repo
        self.revs = list(repo.changelog.revs())
        self.lastrev = self.revs[-1]
        self.firstrev = firstrev
        self.exclude = exclude
        pass
        self.dirs = set()
        self.udirs = set()

    def select(self, paths):
        if len(self.exclude) == 0:
            return paths
        else:
            return filter(lambda _:not(_.startswith(tuple(self.exclude))), paths) 
        pass

    def fileFollow(self, fctx, depth=0):

        if depth == 0:
            self.udirs.clear()

        par = fctx.parents()
        lpar = len(par)

        path = fctx.path()
        fold = os.path.dirname(path)

        self.dirs.add(fold)
        self.udirs.add(fold)

        #print "%4d %2d %30s %s " % (depth,lpar, path, fctx.description())

        if lpar > 0:
            self.fileFollow(par[0], depth+1)


class ChangesetFolderHistory(Follower):
    """
    Follows ancestry of all files within a changeset, 
    optionally restricted by prefix strings collecting 
    the set of folders in which the files have resided. 
    """
    def __init__(self, repo, exclude=[], firstrev=0):
        Follower.__init__(self, repo, exclude, firstrev)
   
    def changeFollow(self, rev):
        cctx = self.repo[rev]
        paths = self.select(cctx.files())
        for f in paths:
            try:
                fctx = cctx[f]
            except:
                # log.warn("failed to lookup %s at rev %s " % (f, rev) )  ## seems to correspons to deletions
                fctx = None

            if fctx is not None:
                self.fileFollow(fctx)



class PathFolderHistory(Follower):
    def __init__(self, repo, exclude=[], firstrev=0):
        Follower.__init__(self, repo, exclude, firstrev)

    def findRev(self, path):
        """
        Looking for a changeset including a path
        """
        for rev in range(self.lastrev, self.firstrev, -1):
            if path in self.repo[rev].files():
                return rev
            pass
        return None
 
    def findRevAndFollow(self, path):
        """
        If a rev is found follow the file history to see where its has been 
        """
        rev = self.findRev(path)
        if rev is None:
            return

        cctx = self.repo[rev]
        try:
            fctx = cctx[path]
        except:
            log.warn("failed to lookup %s at rev %s " % (path, rev) )  ## seems to correspons to deletions
            fctx = None
        pass

        if fctx is not None:
            self.fileFollow(fctx)

        return list(self.udirs)

 
   
class Filemap(object):
    def __init__(self, include, exclude, files="opticks.bash CMakeLists.txt opticksdata.bash bin/op.sh".split()):
        self.include = files + filter(None, sorted(include))
        self.exclude = exclude 

    def __str__(self):
        lines = []
        lines += map(lambda _:"include %s" % _,self.include)
        lines += map(lambda _:"exclude %s" % _,self.exclude)
        return "\n".join(lines)


def opticks_full_history_filemap(repo, firstrev):
    """
    Find the set of folders featuring in the entire histories
    of the files that have changed since first_rev
    
    Too many: 426 dirs
    """
    revs = list(repo.changelog.revs())
    last_rev = revs[-1]

    exclude = ["adm","base","nuwa",]

    cfh = ChangesetFolderHistory(repo, exclude)

    for rev in revs[firstrev:]:
        cfh.changeFollow(rev)
        log.info( "rev %6d dirs %6d " % ( rev, len(cfh.dirs) ))
    pass
    log.info( "%d dirs" % len(cfh.dirs) ) 

    fmp = Filemap(list(cfh.dirs))
    print fmp 

   
       

def opticks_proj_history_filemap(repo, firstrev, dirs, exclude):
    """
    Find the set of folders featuring in the entire history 
    of all files in the list of current opticks project dirs 
    """
    pass
    revs = list(repo.changelog.revs())
    lastrev = revs[-1]
    all_paths = list(repo[lastrev])
    sel_paths = filter(lambda _:_.startswith(tuple(dirs)), all_paths) 

    pfh = PathFolderHistory(repo, exclude=exclude, firstrev=firstrev)

    for path in sel_paths:
        dirs = pfh.findRevAndFollow(path)
        log.info("%50s : %s " % (path, repr(dirs)))
         
    fmp = Filemap(list(pfh.dirs), exclude)
    return fmp 


class EnvOpticks(object):
    dirs_ = r"""
cmake
externals
sysrap
boostrap
opticksnpy
optickscore
ggeo
assimprap
openmeshrap
opticksgeo
oglrap
cudarap
thrustrap
optixrap
opticksop
opticksgl
ggeoview
cfg4
numpyserver
"""
    exclude_ = r"""
externals/DybPolicy
externals/cmt
externals/interfaces
externals/policy
externals/settings
externals/site
cuda/optix/OptiX_301
optix/OptiX_301
cuda/optix/optix301
optix/optix301
"""
    def __init__(self, args):
        self.repopath = os.path.expanduser(args.repopath)
        self.firstrev = args.firstrev 
        self.dirs = self.dirs_.split()
        self.exclude = self.exclude_.split()
        log.info(" repopath %s firstrev %s dirs %s exclude %s " % (self.repopath, self.firstrev, repr(self.dirs), repr(self.exclude))) 
      

def parse(doc):
    parser = argparse.ArgumentParser(doc)
    parser.add_argument(     "--loglevel", default="info")
    parser.add_argument(     "--firstrev", default=4910, type=int )
    parser.add_argument(     "--repopath", default="~/envtmp")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging,args.loglevel.upper()))
    return args 


if __name__ == '__main__':

    args = parse(__doc__)

    eo = EnvOpticks(args)

    repo = hg.repository(ui.ui(), eo.repopath)

    fmp = opticks_proj_history_filemap(repo, eo.firstrev, eo.dirs, eo.exclude)

    print fmp 



