# === func-gen- : hg/hg fgp hg/hg.bash fgn hg fgh hg
hg-src(){      echo hg/hg.bash ; }
hg-source(){   echo ${BASH_SOURCE:-$(env-home)/$(hg-src)} ; }
hg-vi(){       vi $(hg-source) ; }
hg-env(){      elocal- ; }
hg-usage(){
  cat << EOU
Mercurial
===========

Related
--------

* *adm-vi* for conversion from SVN, splitting repos

Tips
----

*hg -v help log*
     gives much more detailed help with the *-v*

*hg log -v -l5*
     *-v* option lists changed files

*hg log -vGl 5*
     combine options, G shows DAG, l to limit revisions 

*hg log --date "2014-05-01 to 2015-04-21"*
     select entries in a date range

merging
~~~~~~~~~~

* https://swcarpentry.github.io/hg-novice/12-merges/


merging with vim
~~~~~~~~~~~~~~~~~~

* https://www.mercurial-scm.org/wiki/MergingWithVim





serve a web interface to quickly check on commits : very handy when get multiple heads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::
 
    epsilon:simoncblyth.bitbucket.io blyth$ hg serve
    listening at http://epsilon.local:8000/ (bound to *:8000)
    10.10.4.175 - - [10/May/2019 11:13:17] "GET / HTTP/1.1" 200 -
    10.10.4.175 - - [10/May/2019 11:13:17] "GET /static/style-paper.css HTTP/1.1" 200 -
    ...


revert without leaving .orig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    # ~/.hgrc
    [ui]
    origbackuppath = /tmp/hg-trash


case wierdness
~~~~~~~~~~~~~~~~~

Capitalized in file system, but not in mercurial ?::

    epsilon:npy blyth$ ll No.*
    -rw-r--r--  1 blyth  staff  1334 Jun 21 14:38 No.cpp
    -rw-r--r--  1 blyth  staff  1070 Jun 21 15:39 No.hpp
    epsilon:npy blyth$ hg revert No.cpp
    no changes needed to no.cpp

    epsilon:npy blyth$ rm no.cpp no.hpp
    epsilon:npy blyth$ hg revert no.cpp
    epsilon:npy blyth$ hg revert no.hpp
    epsilon:npy blyth$ ll No.*
    ls: No.*: No such file or directory
    epsilon:npy blyth$ ll no.*
    -rw-r--r--  1 blyth  staff  1334 Jun 25 13:40 no.cpp
    -rw-r--r--  1 blyth  staff  1070 Jun 25 13:41 no.hpp
    epsilon:npy blyth$ 

    epsilon:npy blyth$ hg rename no.cpp No.cpp
    epsilon:npy blyth$ hg rename no.hpp No.hpp
    epsilon:npy blyth$ hg st .
    M tests/NPYreshapeTest.cc
    A No.cpp
    A No.hpp
    R no.cpp
    R no.hpp
    epsilon:npy blyth$ 



new commits still draft : manually set them public
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    epsilon:opticks-cmake-overhaul blyth$ hg phase 
    2299: draft
    epsilon:opticks-cmake-overhaul blyth$ hg phase -p
    epsilon:opticks-cmake-overhaul blyth$ hg phase 
    2299: public
    epsilon:opticks-cmake-overhaul blyth$ hg push 
    pushing to ssh://hg@bitbucket.org/simoncblyth/opticks-cmake-overhaul
    searching for changes
    no changes found
    epsilon:opticks-cmake-overhaul blyth$ 



hg phases : bitbucket commits showing up DRAFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.mercurial-scm.org/wiki/Phases

Introduction To Mercurial Phases (Part II)

* https://www.logilab.org/blogentry/88219


commandline + bitbucket commits showing up DRAFT 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Manually changed them to public and pushed gets them 
to show up normally on bitbucket.

The repo was pushed to bitbucket from a local clone::

    epsilon:opticks-cmake-overhaul blyth$ hg phase -r 2290:2298
    2290: public
    2291: public
    2292: public
    2293: public
    2294: draft
    2295: draft
    2296: draft
    2297: draft
    2298: draft

    epsilon:opticks-cmake-overhaul blyth$ hg phase --public 2294
    epsilon:opticks-cmake-overhaul blyth$ hg phase -r 2290:2298
    2290: public
    2291: public
    2292: public
    2293: public
    2294: public
    2295: draft
    2296: draft
    2297: draft
    2298: draft
    epsilon:opticks-cmake-overhaul blyth$ hg phase --public 2295
    epsilon:opticks-cmake-overhaul blyth$ hg phase --public 2296
    epsilon:opticks-cmake-overhaul blyth$ hg phase --public 2297
    epsilon:opticks-cmake-overhaul blyth$ hg phase --public 2298
    epsilon:opticks-cmake-overhaul blyth$ hg phase -r 2290:2298
    2290: public
    2291: public
    2292: public
    2293: public
    2294: public
    2295: public
    2296: public
    2297: public
    2298: public
    epsilon:opticks-cmake-overhaul blyth$ 

    epsilon:opticks-cmake-overhaul blyth$ hg push 
    pushing to ssh://hg@bitbucket.org/simoncblyth/opticks-cmake-overhaul
    searching for changes
    no changes found
    epsilon:opticks-cmake-overhaul blyth$ 


hg equivalent of git push -u
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   vi .hg/hgrc 
    
Then edit the paths:: 

  2 [paths]
  3 default = ssh://hg@bitbucket.org/simoncblyth/opticks-cmake-overhaul
  4 #default = /Users/blyth/opticks



merging opticks-cmake-overhaul back into opticks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* there are no commits in opticks, all commits done into opticks-cmake-overhaul
  over the past 6 weeks

In brief::

   cd ~/opticks
   hg pull ../opticks-cmake-overhaul
   hg update
   hg outgoing   ## shows what will push, about 100 commits from the past 6 weeks
   hg push       ## 98 changesets with 2274 changes to 1223 files


::

    epsilon:opticks blyth$ hg pull ../opticks-cmake-overhaul
    pulling from ../opticks-cmake-overhaul
    searching for changes
    adding changesets
    adding manifests
    adding file changes
    added 98 changesets with 2274 changes to 1223 files
    new changesets 92cfcc4149af:24f381236afe
    (run 'hg update' to get a working copy)
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ hg update
    1164 files updated, 0 files merged, 474 files removed, 0 files unresolved
    epsilon:opticks blyth$ 


    epsilon:opticks blyth$ hg outgoing   ## shows what will push, about 100 commits from the past 6 weeks
    comparing with ssh://hg@bitbucket.org/simoncblyth/opticks
    searching for changes
    changeset:   2294:92cfcc4149af
    user:        Simon Blyth <simoncblyth@gmail.com>
    date:        Wed May 16 21:03:57 2018 +0800
    summary:     exploring approaches to modern CMake with exported/imported targets, and using bcm- to avoid the boilerplate

    changeset:   2295:351c13fb857a
    user:        Simon Blyth <simoncblyth@gmail.com>
    date:        Thu May 17 19:31:52 2018 +0800
    summary:     finding a way use my BCM fork with unchanged Opticks header locations in source and installation, so far works in the simple UseGLMViaBCM example

    ...

    changeset:   2390:2e86e80d2d37
    user:        Simon Blyth <simoncblyth@gmail.com>
    date:        Mon Jun 25 13:43:49 2018 +0800
    summary:     preparing to bring developments from opticks-cmake-overhaul back home to opticks using transient opticks-test-copy to check first

    changeset:   2391:24f381236afe
    tag:         tip
    user:        Simon Blyth <simoncblyth@gmail.com>
    date:        Mon Jun 25 14:01:21 2018 +0800
    summary:     move opticks-full to use om-install


::

    epsilon:opticks blyth$ hg push 
    pushing to ssh://hg@bitbucket.org/simoncblyth/opticks
    searching for changes
    remote: adding changesets
    remote: adding manifests
    remote: adding file changes
    remote: added 98 changesets with 2274 changes to 1223 files
    epsilon:opticks blyth$ 



opticks-cmake-overhaul
~~~~~~~~~~~~~~~~~~~~~~~~

Overhauling CMake infrastructure is bound to cause 
build breakage potentially for an extended period, 
so are unwilling to commit/push the CMake related changes.

Instead: 

1. commit/push all unrelated non-breaking changes are willing to, leaving 
   just the CMake related ones::

::

    epsilon:opticks blyth$ hg st .
    M CMakeLists.txt
    M cmake/Templates/opticks-config.in
    M okop/okop.bash
    M opticks.bash
    M opticksnpy/CMakeLists.txt
    M sysrap/CMakeLists.txt
    ? cmake/Modules/FindOpticks.cmake
    ? cmake/Modules/OpticksConfigureCMakeHelpers.cmake
    ? cmake/Templates/OpticksConfig.cmake.in
    ? examples/FindOpticks/CMakeLists.txt
    ? examples/FindOpticks/FindOpticks.cc
    ? examples/FindOpticks/README.rst
    ? examples/FindOpticks/go.sh
    ? examples/UseNPY/CMakeLists.txt
    ? examples/UseNPY/UseNPY.cc
    ? examples/UseNPY/go.sh
    ? examples/UseSysRap/CMakeLists.txt
    ? examples/UseSysRap/UseSysRap.cc
    ? examples/UseSysRap/go.sh
    epsilon:opticks blyth$ 

2. make a local clone::

    cd ; hg clone opticks opticks-cmake-overhaul    ## apparently this uses hardlinks

    epsilon:opticks-cmake-overhaul blyth$ hg paths -v    ## can pull/update from "mainline" into the overhaul clone 
    default = /Users/blyth/opticks

    epsilon:opticks blyth$ mv examples ../opticks-cmake-overhaul/


3. caveman move CMake development over to the local overhaul clone::

    epsilon:opticks blyth$ hg st .
    M CMakeLists.txt
    M cmake/Templates/opticks-config.in
    M okop/okop.bash
    M opticks.bash
    M opticksnpy/CMakeLists.txt
    M sysrap/CMakeLists.txt
    ? cmake/Modules/FindOpticks.cmake
    ? cmake/Modules/OpticksConfigureCMakeHelpers.cmake
    ? cmake/Templates/OpticksConfig.cmake.in
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ mv cmake/Modules/FindOpticks.cmake ../opticks-cmake-overhaul/cmake/Modules/
    epsilon:opticks blyth$ mv cmake/Modules/OpticksConfigureCMakeHelpers.cmake ../opticks-cmake-overhaul/cmake/Modules/
    epsilon:opticks blyth$ mv cmake/Templates/OpticksConfig.cmake.in ../opticks-cmake-overhaul/cmake/Templates/
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ cp CMakeLists.txt ../opticks-cmake-overhaul/
    epsilon:opticks blyth$ cp cmake/Templates/opticks-config.in ../opticks-cmake-overhaul/cmake/Templates/
    epsilon:opticks blyth$ cp okop/okop.bash ../opticks-cmake-overhaul/okop/
    epsilon:opticks blyth$ cp opticks.bash ../opticks-cmake-overhaul/
    epsilon:opticks blyth$ cp opticksnpy/CMakeLists.txt ../opticks-cmake-overhaul/opticksnpy/
    epsilon:opticks blyth$ cp sysrap/CMakeLists.txt ../opticks-cmake-overhaul/sysrap/
    epsilon:opticks blyth$ 

4. revert the changes and rm .orig to return to pristine opticks::  

    epsilon:opticks blyth$ hg revert CMakeLists.txt
    epsilon:opticks blyth$ hg revert cmake/Templates/opticks-config.in
    epsilon:opticks blyth$ hg revert okop/okop.bash
    epsilon:opticks blyth$ hg revert opticks.bash
    epsilon:opticks blyth$ hg revert opticksnpy/CMakeLists.txt
    epsilon:opticks blyth$ hg revert sysrap/CMakeLists.txt


5. check opticks still builds, and the tests run::

   opticks--
   opticks-t   ## one (1/300) familiar fail from GSceneTest


opticks-cmake-overhaul
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. pick via .bash_profile::

    308 #export OPTICKS_HOME=$HOME/opticks
    309 export OPTICKS_HOME=$HOME/opticks-cmake-overhaul




branches
~~~~~~~~~~~

* http://stevelosh.com/blog/2009/08/a-guide-to-branching-in-mercurial/

When you commit the newly created changeset will be on the same branch as its
parent, unless you’ve used hg branch to mark it as being on a different one.

bookmarks
~~~~~~~~~~~

* https://www.mercurial-scm.org/wiki/Bookmarks

Mercurial's bookmark feature is analogous to Git's branching scheme, but can
also be used in conjunction with Mercurial's traditional named branches.

* http://mercurial.aragost.com/kick-start/en/bookmarks/

Tags also add new names to existing changesets, but unlike tags, bookmarks are
mutable and transient: you can move, rename, or delete them and they are not
stored in history. This means that there is no audit trail for bookmarks.

::

    hg bookmarks   ## list bookmarks

The asterisk (*) indicates that the bookmark is active, which means that it
will move along if she makes a new commit. Because the active bookmark moves
along when you commit, it will always point to the head of the branch you’re
working on.



mercurial branch merge workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.mercurial-scm.org/wiki/Workflows




log templates
~~~~~~~~~~~~~~

* http://stackoverflow.com/questions/3575189/mercurial-log-with-one-liners

Many examples of templates at http://www.selenic.com/hg/help/templates


Add log alias to .hgrc::

    [alias]
    shortlog = log --template '{node|short} | {date|isodatesec} | {author|user}: {desc|strip|firstline}\n'
    flog = log --template "\n{node|short} | {date|isodatesec} | {author|user}: {desc|strip|firstline}\n{files % '  {file}\n'}"

Then can review period by period with::

    hg shortlog --date "2014-05-01 to 2014-05-07" | tail -r



Start using new node
~~~~~~~~~~~~~~~~~~~~~~

#. get mercurial, git, svn and cmake installed 

   * if mercurial not installed and do not have root access
     then use mercurial- for a local install


Copy/paste public key into bitbucket webinterface::

    (chroma_env)delta:.ssh blyth$ scp G5:.ssh/id_dsa.pub G5.id_dsa.pub
    (chroma_env)delta:.ssh blyth$ cat G5.id_dsa.pub | pbcopy 

Copy identity config to node::

    delta:~ blyth$ scp ~/.hgrc G5:
    delta:~ blyth$ scp .hgrc L:

Clone took more than 5 mins::

    [blyth@ntugrid5 ~]$ hg clone ssh://hg@bitbucket.org/simoncblyth/env
    destination directory: env
    requesting all changes
    adding changesets
    adding manifests
    adding file changes
    added 4737 changesets with 14682 changes to 4602 files
    updating to branch default
    3166 files updated, 0 files merged, 0 files removed, 0 files unresolved
    [blyth@ntugrid5 ~]$ 


Start with new Windows/NSYS2 node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After NSYS2 and env setup::

  ssh--keygen  # enter passphase and document it in normal place

  cat id_rsa.pub | clip    # clip is windows equivalent of pbcopy

  login to bitbucket web interface using windows chrome browser
 
  right click paste into the web form for the key


Switch default path from http to ssh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After cloning over http the env/.hg/hgrc defaults to http, switch that to ssh::

   [paths]
   #default = http://bitbucket.org/simoncblyth/env
   default = ssh://hg@bitbucket.org/simoncblyth/env

Also check that ~/.hgrc contains::

  [ui]
  username = Simon Blyth <simoncblyth@gmail.com>
  ssh = ssh -C


SSH Setup on windows
~~~~~~~~~~~~~~~~~~~~~

* https://confluence.atlassian.com/bitbucket/set-up-ssh-for-mercurial-728138122.html
* https://confluence.atlassian.com/bitbucket/set-up-ssh-for-git-728138079.html 

Bitbucket/Mercurial instructions entail installing putty and diddling with
other GUI type applications, nasty.  

Bitbucket/Git instructions use the git bash shell that comes with git-for-windows, so 
do the ssh setup using the git-for-windows approach and can then use the keys 
for mercurial. 

This succeeds to provide passwordless push/pull from gitbash, but not from powershell. Get::

   remote: 'ssh' is not recognized as an internal or external command



Passwordless Operations
~~~~~~~~~~~~~~~~~~~~~~~~~

Start and authenticate agent::

   ssh--agent-start



Rollback a commit before a push 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Annoyingly Mercurial will plow ahead with a commit even after not finding any username
configured.  This usually results in a bad username, causes the commit to appear 
as anonymous in Bitbucket::

    [blyth@cms01 env]$ hg st 
    M base/digestpath.py
    [blyth@cms01 env]$ hg commit -m "add dnatree findoption of for example -L, not changing behaviour yet " 
    No username found, using 'blyth@cms01.phys.ntu.edu.tw' instead

        ## rollback is dangerous and not recommended, but using older mercurial on C so commit has no "--amend" option 

    [blyth@cms01 env]$ hg rollback 
    rolling back last transaction
    [blyth@cms01 env]$ hg st 
    M base/digestpath.py
    [blyth@cms01 env]$ 

Copy the Mercurial config from another node::

    delta:~ blyth$ scp ~/.hgrc C:

Verify the username before committing::

    [blyth@cms01 env]$ hg showconfig   
    bundle.mainreporoot=/home/blyth/env
    extensions.hgext.convert=
    paths.default=ssh://hg@bitbucket.org/simoncblyth/env
    ui.ssh=ssh -C
    ui.username=Simon Blyth <simoncblyth@gmail.com>


Aborted push 
~~~~~~~~~~~~~

To avoid the below rigmarole, try to operate
in pass-the-baton fashion as move from machine to machine
and always push/pull as pass baton.

A push is aborted::

    delta:env blyth$ hg commit -m "ruminate on how to integrate chroma formed GPU hits into the flow " 
 
    delta:env blyth$ hg push 
    pushing to ssh://hg@bitbucket.org/simoncblyth/env
    searching for changes
    abort: push creates new remote head 658f4429167b!
    (pull and merge or see "hg help push" for details about pushing new heads)


Pulling changes gets into multi-headed state::

    delta:env blyth$ hg pull
    pulling from ssh://hg@bitbucket.org/simoncblyth/env
    searching for changes
    adding changesets
    adding manifests
    adding file changes
    added 1 changesets with 1 changes to 1 files (+1 heads)
    (run 'hg heads' to see heads, 'hg merge' to merge)
    delta:env blyth$ 

    delta:env blyth$ hg heads
    changeset:   4705:2b3f791f9c74
    tag:         tip
    parent:      4703:57625e9292d1
    user:        Simon Blyth <simoncblyth@gmail.com>
    date:        Tue Oct 14 19:59:28 2014 +0800
    summary:     csa generalize

    changeset:   4704:658f4429167b
    user:        Simon Blyth <simoncblyth@gmail.com>
    date:        Tue Oct 14 21:02:13 2014 +0800
    summary:     ruminate on how to integrate chroma formed GPU hits into the flow


Merge brings them together, with uncommited changes::

    delta:env blyth$ hg merge
    1 files updated, 0 files merged, 0 files removed, 0 files unresolved
    (branch merge, don't forget to commit)
    delta:env blyth$ 
    delta:env blyth$ hg st 
    M nuwa/detsim/csa.bash
    delta:env blyth$ 





Simple Mercurial Backup Procedure
-----------------------------------

One time setup
~~~~~~~~~~~~~~~

#. *hg init* repository on remote node that have ssh keyed access to::

        [blyth@cms01 mercurial]$ pwd
        /data/var/scm/mercurial
        [blyth@cms01 mercurial]$ hg init env

#. on working node edit the .hg/hgrc paths section adding a *backup* alias
   specifying *user@host//path* 
   The double slash signifies an absolute path, a single slash would be relative to HOME.::

       delta:env blyth$ hg paths
       default = ssh://hg@bitbucket.org/simoncblyth/env
       backup = ssh://blyth@cms01.phys.ntu.edu.tw//data/var/scm/mercurial/env



Manual Backup
~~~~~~~~~~~~~~

After commiting local changes and pushing to bitbucket, occasionally make
an extra backup::

    cd ~/env
    hg push backup
    hg-check


Repo digest
~~~~~~~~~~~~

Repository digests are computed from file content of the .hg/store directory 
excluding two files that were found to differ depending on the architecture

* .hg/store/fncache 
* .hg/store/undo

For details on Mercurial repo format 

* http://mercurial.selenic.com/wiki/FileFormats
* http://mercurial.selenic.com/wiki/fncacheRepoFormat

    (adm_env)delta:env blyth$ hg push backup
    pushing to ssh://blyth@cms01.phys.ntu.edu.tw/../../data/var/scm/mercurial/env
    remote: Scientific Linux CERN SLC release 4.8 (Beryllium)
    searching for changes
    remote: adding changesets
    remote: adding manifests
    remote: adding file changes
    remote: added 4635 changesets with 14025 changes to 4361 files



Failed Repo Digest Approaches
--------------------------------

bundle digests
~~~~~~~~~~~~~~~~

#. pushed backup bundle has different md5sum

::

    (adm_env)delta:env blyth$ hg bundle --all env.bundle
    4635 changesets found
    (adm_env)delta:env blyth$ md5 env.bundle
    MD5 (env.bundle) = c3bdc095c43af222ffe87cbb11ee50eb

tarred digest
~~~~~~~~~~~~~~~~~~

::

    delta:env blyth$ ( cd /tmp/t/env ; tar -cf - --exclude .hg/hgrc .hg ) | md5
    d1f55c3d8beeb2ddb66bdbb54f7bb03a

    delta:env blyth$ ( cd /tmp/s/env ; tar -cf - --exclude .hg/hgrc .hg ) | md5
    f80ae809ecf9352c376a143da52b045e


leaf digests
~~~~~~~~~~~~~~

Excluding two top level files succeeds to get a digest match for two clones
both on OSX::

    delta:env blyth$ digest.py /tmp/t/env/.hg hgrc dirstate 
    3e7d9e26fefdfeaba6e5facf8f68148b

    delta:env blyth$ digest.py /tmp/s/env/.hg hgrc dirstate 
    3e7d9e26fefdfeaba6e5facf8f68148b

Comparing a repo on OSX with a pushed backup on Linux, differs
apparently due to a file naming difference problem scrambling the digest ordering::

    [blyth@cms01 env]$ l .hg/store/data/base/.ssh.bash.swp.i 
    -rw-rw-r--  1 blyth blyth 1560 Aug 29 17:53 .hg/store/data/base/.ssh.bash.swp.i

    (adm_env)delta:env blyth$ l .hg/store/data/base/~2essh.bash.swp.i 
    -rw-r--r--  3 blyth  staff  1560 Aug 29 16:46 .hg/store/data/base/~2essh.bash.swp.i

After fixing the file ordering by working around the above name problem 
succeed to get a match within the store:: 

    (adm_env)delta:env blyth$ digest.py .hg/store fncache undo
    15fdcc096e0252f4a6ee9ed12b86005a

    [blyth@cms01 env]$ digest.py .hg/store fncache undo
    15fdcc096e0252f4a6ee9ed12b86005a

Suspect getting a complete .hg match liable to be impossible without using 
precisely the same Mercurial version at both ends.




hg convert
------------

* see adm- for higher level usage with full history comparisons


#. needs *sudo port install subversion-python27bindings*
#. authormap
#. timezone

* http://hgbook.red-bean.com/read/migrating-to-mercurial.html


env repo took about 8 minutes over network to D, 1 min with local mirror
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    delta:t blyth$ hg convert --source-type svn --dest-type hg  http://dayabay.phys.ntu.edu.tw/repos/env/trunk envhg
    initializing destination envhg repository
    scanning source...

Create a bare repo .hg running hg serve and browsing the html reveals issues:

#. timezone, UTC is obnoxious unless can localize 
#. authormap


#. the bare repo should be admin owned ? 
#. *hg-convert* DOES work incrementally, after the convert 
   which gets new revisions from SVN into the bare repo,
   *hg pull* changes from the bare repo into the working repo 
   and then *hg update* into working copy


wc comparison with svn
~~~~~~~~~~~~~~~~~~~~~~~~

::

    delta:env blyth$ hg-compare-with-svn 
    diff -r --brief /Users/blyth/env /tmp/mercurial/env
    Only in /tmp/mercurial/env: .hg
    Only in /Users/blyth/env: .svn
    Only in /Users/blyth/env: _build   # THIS COULD BE MOVED ELSEWHERE
    delta:env blyth$ 

systematic history checking 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://svn.apache.org/repos/asf/subversion/trunk/tools/examples/
* hgapi

Operational Curiosities
-------------------------

divergent renames
~~~~~~~~~~~~~~~~~~~

This warning from 2014-09-24 seems to have no bad consequences::

    blyth@belle7 env]$ hg up
    warning: detected divergent renames of muon_simulation/presentation/gpu_optical_photon_simulation.txt to:
     presentation/g4dae_geometry_exporter.txt
     presentation/gpu_optical_photon_simulation.txt
    115 files updated, 0 files merged, 88 files removed, 0 files unresolved


FUNCTIONS
---------

hg-repos <dir:PWD> :
     list absolute paths of repos beneath dir, defaulting to PWD

hg-vals default
     list .hg/hgrc metadata 

hg-pull 
     pull into all repos found beneath PWD

hg-backup
     effect the backup of remote repos by pulling into local backup
     to add another repo to the backup initiate manual clone
     into the backup dir, eg::

         cd /var/hg/backup
         hg clone http://belle7.nuu.edu.tw/hg/AuthKitPy24

hg-forest-get
     openjdk- requires forrest extenstion to mercurial
     macports gets dependencies py2.7.2 and mercurial 2.1

     initially "hg help" shows no sign of forrect, need to 
	 configure extension in ~/.hgrc with::

		  [extensions]
		  hgext.forest=


hg-convert
     * http://mercurial.selenic.com/wiki/ConvertExtension

     #. works incrementally, just converting changes
     #. note only trunk is being converted, this is fine for: 
        env, heprez, workflow
     #. tracdev adopts a multiple trunks under toplevel folders
        layout which will need special file mapping 



EOU
}
hg-dir(){ echo /var/hg ;  }
hg-cd(){  cd $(hg-dir); }
hg-mate(){ mate $(hg-dir) ; }

hg-get(){
   local msg="=== $FUNCNAME :"
   [ "$(which hg)" != "" ] && echo $msg hg is already avilable && return 0
   [ "$(which python)" == "/usr/bin/python" ] && echo $msg when using system python its best to use system mercurial, install with yum etc.. && return 0  

   #easy_install -U mercurial   seems I am using the macports one
}

hg-forest-get(){
    sudo port -v install hg-forest

}



hg-par(){ hg parent --template '{node}' ; }  ## full hash of current update
hg-id(){ hg --debug id -i ; }                ## lastest update longhash  

hg-ll(){ hg log -l1 ; }
hg-timestamp(){ TZ=UTC $FUNCNAME- ; }    ## suspect log -l1 is latest in repo, not necessary the current one
hg-timestamp-(){ hg log -l1 --template '{date(date|localdate, "%c")}\n' ; }

hg-year(){ echo ${HG_YEAR:-$(date +"%Y")} ; }

hg-18(){ HG_YEAR=2018 hg-month $* ; }
hg-17(){ HG_YEAR=2017 hg-month $* ; }
hg-16(){ HG_YEAR=2016 hg-month $* ; }
hg-15(){ HG_YEAR=2015 hg-month $* ; }
hg-14(){ HG_YEAR=2014 hg-month $* ; }
hg-13(){ HG_YEAR=2013 hg-month $* ; }


hg-month(){
   # negate the month argument for prior years month 

   local arg=${1:-12}
   local year=$(hg-year) 

   [ "${arg:0:1}" == "-" ] && arg=${arg:1} && year=$(( $year - 1))

   local beg=$(printf "%0.2u" $arg)
   local end=$(printf "%0.2u" $(( $arg + 1)))

   local byear=$year
   local eyear=$year

   [ "${end}" == "13" ] && end="01" && eyear=$(( $byear + 1  ))


   local cmd="hg shortlog --date \"$byear-$beg-01 to $eyear-$end-01\" "
   case $(uname) in
      Darwin) cmd="$cmd | tail -r" ;;
      Linux)  cmd="$cmd | tac" ;;
   esac
   echo $cmd
   eval $cmd
}



hg-repos(){ find ${1:-$PWD} -type d -name '.hg' -exec dirname {} \;; }
hg-val(){   perl -n -e "m/^$2\s*=\s*(.*)$/ && print \$1 " $1/.hg/hgrc ; }
hg-vals(){
   local v=${1:-default}
   local repo
   hg-repos | while read repo ; do
      echo $repo $v $(hg-val $repo $v)
   done
}
hg-pull(){
   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   local repo
   hg-repos | while read repo ; do
      cd $repo
      echo $msg pulling into $PWD from $(hg-val $repo default)
      hg pull
   done
   cd $iwd
}

hg-pull-backup(){
   local dir=$(hg-dir)/backup 
   mkdir -p $dir && cd $dir
   hg-pull
}

hg-repodir(){ echo /var/scm/mercurial/${1:-env} ; }
hg-wcdir(){   echo /tmp/mercurial/${1:-env} ; }

hg-convert(){
   local msg="=== $FUNCNAME :"
   local repo=${1:-env}
   local hgr=$(hg-repodir $repo)
   local url=http://dayabay.phys.ntu.edu.tw/repos/$repo/trunk

  # it does work incrementally, no need to start from scratch 
  # [ -d "$hgr" ] && echo $msg PREEXISTING hgr $hgr : DELETE THIS AND TRY AGAIN && return

   [ ! -d "$hgr" ] && echo $msg creating $hgr && mkdir -p $hgr

   local cmd="hg convert --config convert.localtimezone=true --source-type svn --dest-type hg $url $hgr"
   echo $cmd
   date
   eval $cmd
   date
}

hg-find-empty(){
   find . -mindepth 1 -type d -empty
}

hg-clone(){
   local repo=${1:-env}
   local hgr=$(hg-repodir $repo)
   local dir=$(hg-wcdir $repo)
   local base=$(dirname $dir)
 
   mkdir -p $(hg-wcdir)
   cd $base
   hg clone $hgr   
   # is this the recommended approach, having a bare repo and cloning to where you work ? 
}


hg-compare-with-svn(){
   local repo=${1:-env}
   local cmd="diff -r --brief $HOME/$repo $(hg-wcdir $repo)"
   echo $cmd
   eval $cmd


}


hg-backup(){
    hg push backup
    hg-check
}
hg-check(){
    local msg="=== $FUNCNAME : "
    echo $msg Comparing stores : $PWD $(hg path backup)
    hg-cdig
}
hg-cdig(){
    local msg="=== $FUNCNAME : "
    local ldig=$(echo $(hg-ldig))
    local rdig=$(echo $(hg-rdig))

    echo $msg LOC $ldig
    echo $msg REM $rdig

    if [ "$ldig" != "$rdig" ]; then 
        echo $msg LOCAL/REMOTE DIGEST MISMATCH ldig $ldig rdig $rdig
        sleep 100000000
    else
        echo $msg DIGEST MATCH OK 
    fi 
}
hg-ldig(){
    digest.py .hg/store fncache undo
}
hg-rdig(){
    local bkp=$(hg path backup)
    local rpath=$(echo ${bkp/*\/\/})
    local nhp=$(echo ${bkp/ssh:\/\/})  # name@host//path 
    local namehost=$(echo ${nhp/\/\/*})
    #
    ssh $namehost "bash -lc 'digest.py /$rpath/.hg/store fncache undo'" 2>/dev/null
}

