# === func-gen- : git/git fgp git/git.bash fgn git fgh git
git-src(){      echo git/git.bash ; }
git-source(){   echo ${BASH_SOURCE:-$(env-home)/$(git-src)} ; }
git-vi(){       vi $(git-source) ; }
git-env(){      elocal- ; }
git-usage(){ cat << EOU

Git
====

See Also
----------

* github-


Reference
------------

* http://gitref.org/remotes/#fetch
* http://book.git-scm.com/book.pdf 
* http://www.git-scm.com/book/en/Git-Basics-Getting-a-Git-Repository

* https://www.rath.org/mercurial-for-git-users-and-vice-versa.html

Git Tags

Good Reference
-----------------

* https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts
* https://www.atlassian.com/git/tutorials/comparing-workflows
* https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow


Git Help
----------

::

    git help cli


Git Dry Run Merge
--------------------

* https://stackoverflow.com/questions/501407/is-there-a-git-merge-dry-run-option
* https://www.janbasktraining.com/community/devops/is-there-a-git-merge-dry-run-option


Do the merge, but dont commit, and dont allow ff (as commit couldnt then be stopped)::

    git merge --no-commit --no-ff $branch
    ## exit status is 0 iff the merge is possible.

Examine staged changes::

    git diff --cached
    ## "--cached" is synonym for "--staged"

Undo merge::

    git merge --abort



Remove the last commit (only sensible when you did not push it yet) : git reset --hard HEAD^
-----------------------------------------------------------------------------------------------

::

    epsilon:junosw_check2 blyth$ git log -n2
    commit 50fdede6466db384a5b970bf94a0ddc44cf6ab35 (HEAD -> blyth-88-pivot-PMT-optical-model-from-FastSim-to-CustomG4OpBoundaryProcess)
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Wed Apr 19 14:39:30 2023 +0100

        relocate setup_generator_opticks in attempt to avoid merge conflict

    commit 0b8f5cbf46079ac8366f687d898f760865485f44
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Tue Apr 18 20:12:26 2023 +0100

        chomp the eol at end of cmake/JUNODependencies.cmake as seems to cause merge conflict
    epsilon:junosw_check2 blyth$ 
    epsilon:junosw_check2 blyth$ git status 
    On branch blyth-88-pivot-PMT-optical-model-from-FastSim-to-CustomG4OpBoundaryProcess
    Your branch is ahead of 'origin/blyth-88-pivot-PMT-optical-model-from-FastSim-to-CustomG4OpBoundaryProcess' by 2 commits.
      (use "git push" to publish your local commits)

    nothing to commit, working tree clean
    epsilon:junosw_check2 blyth$ 
    epsilon:junosw_check2 blyth$ 
    epsilon:junosw_check2 blyth$ git reset --hard HEAD^
    HEAD is now at 0b8f5cb chomp the eol at end of cmake/JUNODependencies.cmake as seems to cause merge conflict
    epsilon:junosw_check2 blyth$ 
    epsilon:junosw_check2 blyth$ 
    epsilon:junosw_check2 blyth$ git log -n2
    commit 0b8f5cbf46079ac8366f687d898f760865485f44 (HEAD -> blyth-88-pivot-PMT-optical-model-from-FastSim-to-CustomG4OpBoundaryProcess)
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Tue Apr 18 20:12:26 2023 +0100

        chomp the eol at end of cmake/JUNODependencies.cmake as seems to cause merge conflict

    commit a6cded9e334595ca47a6f878ffb100bfd45d4b3c (origin/blyth-88-pivot-PMT-optical-model-from-FastSim-to-CustomG4OpBoundaryProcess)
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Tue Apr 18 16:34:20 2023 +0100

        try changing JUNOTOP to J23.1.x in .gitlab-ci.yml in attempt to allow the CI runners to find Custom4, as suggested by Tao in MR 180
    epsilon:junosw_check2 blyth$ 




Rebase vs Merge
------------------

* https://www.atlassian.com/git/tutorials/merging-vs-rebasing

* https://www.atlassian.com/git/tutorials/rewriting-history/git-rebase#:~:text=What%20is%20git%20rebase%3F,of%20a%20feature%20branching%20workflow.

Rebase is one of two Git utilities that specializes in integrating changes from
one branch onto another. The other change integration utility is git merge.
Merge is always a forward moving change record. Alternatively, rebase has
powerful history rewriting features. 


Danger of Rebase : because its rewriting history
--------------------------------------------------

* https://medium.com/devops-with-valentine/gitlab-merge-blocked-fast-forward-merge-is-not-possible-7f86bf79e58b


see what you have been working on in the past 10 commits
-----------------------------------------------------------

::

    epsilon:junosw blyth$ git diff --name-only @~10
    Examples/Tutorial/python/Tutorial/JUNODetSimModule.py
    Simulation/DetSimV2/DetSimMTUtil/src/DetFactorySvc.cc
    Simulation/DetSimV2/DetSimOptions/include/LSExpDetectorConstruction_Opticks.hh
    Simulation/DetSimV2/DetSimOptions/src/DetSim0Svc.cc
    Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction_Opticks.cc
    Simulation/DetSimV2/PMTSim/PMTSim/junoSD_PMT_v2_Debug.h
    Simulation/DetSimV2/PMTSim/include/junoPMTOpticalModel.hh
    Simulation/DetSimV2/PMTSim/include/junoSD_PMT_v2.hh
    Simulation/DetSimV2/PMTSim/include/junoSD_PMT_v2_Opticks.hh
    Simulation/DetSimV2/PMTSim/src/HamamatsuR12860PMTManager.cc
    Simulation/DetSimV2/PMTSim/src/NNVTMCPPMTManager.cc
    Simulation/DetSimV2/PMTSim/src/PMTSDMgr.cc
    Simulation/DetSimV2/PMTSim/src/junoPMTOpticalModel.cc
    Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2.cc
    Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2_Opticks.cc
    Simulation/DetSimV2/PhysiSim/include/DsG4Scintillation.h
    Simulation/DetSimV2/PhysiSim/include/DsPhysConsOptical.h
    Simulation/DetSimV2/PhysiSim/src/DsG4Scintillation.cc
    Simulation/DetSimV2/PhysiSim/src/DsPhysConsOptical.cc
    epsilon:junosw blyth$ 




diff between head and the prior commit
----------------------------------------------

::

    epsilon:junosw blyth$ git diff @~ Simulation/SimSvc/PMTSimParamSvc/src/PMTSimParamSvc.cc


diff between head and an earlier commit
-----------------------------------------

::

    epsilon:junosw blyth$ git diff --name-status @~6
    M       Examples/Tutorial/python/Tutorial/JUNODetSimModule.py
    M       Simulation/DetSimV2/DetSimMTUtil/src/DetFactorySvc.cc
    M       Simulation/DetSimV2/DetSimOptions/src/DetSim0Svc.cc
    M       Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction_Opticks.cc
    M       Simulation/DetSimV2/PMTSim/PMTSim/junoSD_PMT_v2_Debug.h
     


difftool
----------

* https://borgs.cybrilla.com/tils/opendiff-as-difftool/

~/.gitconfig::

    [diff]
        tool = opendiff

    [difftool]
      prompt = false

    [difftool "opendiff"]
        cmd = /usr/bin/opendiff \"$LOCAL\" \"$REMOTE\" -merge \"$MERGED\" | cat


Unsure of the need/reason for the third stanza. 


::

    git difftool  main..$branch -- Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2.cc


* opendiff (aka FileMerge) is macOS GUI diff tool 



Getting branch uptodate
-------------------------

https://stackoverflow.com/questions/20101994/how-to-git-pull-from-master-into-the-development-branch

::

    git fetch origin          ## 
    git checkout master
    git merge --ff-only origin/master
    git checkout dmgr2
    git merge --no-ff origin/master





git log without paging and limiting the number of commits
------------------------------------------------------------

::

    git --no-pager l -n 5


git log 
----------

::

    git --no-pager log 636e^.. --pretty=oneline 


git log/diff : between two commits
------------------------------------------------

::

    git log 98a8d81..4facbde 
    git diff 98a8d81..4facbde 

    git diff 4facbde^-1
       # shorthand for diff between commit and its parent ? 
       # makes sense for simple commits, unclear with merge commits as 2 parents?
 
    git diff 5e0cb90^-1
       # doesnt work for initial commit as there us no parent 


git checkout 98a8d81 : back to prior commit in detached HEAD state
--------------------------------------------------------------------

Return to normal with::

   git checkout main
   git checkout master


git clone : locally 
---------------------

::

    git clone existing_repo_dir new_repo_dir

But this looses the remote info, so better for clone repo of interest on github first 
and then clone that to laptop. 
  


git show : to look at a file from a different branch or at earlier commit
----------------------------------------------------------------------------

::

    epsilon:junosw blyth$ git show main:Simulation/DetSimV2/PMTSim/src/junoSD_PMT_v2.cc > /tmp/conflict/junoSD_PMT_v2.cc 

    git show 98a8d8:ImagePreview/DestinationView.swift 


git archive : extract distrib archive without repo metadata/history
---------------------------------------------------------------------

Looses git history, starting fresh from some commit. 
Great for speculative investigations:: 

    # create zip archive of some commit  
    git archive -o /tmp/98a8d81.zip 98a8d81 
    unzip -l /tmp/98a8d81.zip

    # create reponame in bitbucket web interface 
    cd
    git clone git@bitbucket.org:simoncblyth/reponame.git   # clone the empty 

    unzip /tmp/98a8d81.zip -d reponame                     # populate from zip

    git add . 
    git commit -m "initial commit from ..."
    git push 


 

git show : quick look at earlier versions without needing to checkout
-------------------------------------------------------------------------

::

    epsilon:opticks blyth$ git log -n10 qudarap/QPMT.hh 
    epsilon:opticks blyth$ git show f464a81:qudarap/QPMT.hh
    epsilon:opticks blyth$ git show a005d53:qudarap/QPMT.hh



~/.gitconfig global aliases
------------------------------

::

     10 [alias]
     11     lg = log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit
     12     lga = log --graph --pretty=format:'%h -%d %s (%cr) <%an>' --abbrev-commit
     13     l = log --name-status
     14     ls = log --stat
     15     s = status
     16     ignore = "!gi() { curl -L -s https://www.gitignore.io/api/$@ ;}; gi"



use the alias : git l be787bf3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    epsilon:opticks blyth$ git l be787bf3
    commit be787bf3d6784e946d0e116916636a86cfbd266b
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Tue Oct 6 19:00:13 2020 +0100

        std::vector erase between iterators needs newer GCC version for compilation, so protect it with preprocessor macro

    M       sysrap/tests/SVecTest.cc



git reset HEAD path : to unstage some changes, but leave the changes intact 
------------------------------------------------------------------------------

After accidentally adding some changes::

    epsilon:numpyserver blyth$ o
    On branch master
    Your branch is up-to-date with 'origin/master'.

    Changes to be committed:
      (use "git reset HEAD <file>..." to unstage)

        modified:   boostrap/BCfg.cc
        modified:   boostrap/BCfg.hh
        modified:   numpyserver/numpydelegate.cpp
        modified:   numpyserver/numpyserver.hpp
        modified:   oglrap/OpticksViz.cc
        modified:   oglrap/OpticksViz.hh
        ...

    Changes not staged for commit:
      (use "git add <file>..." to update what will be committed)
      (use "git checkout -- <file>..." to discard changes in working directory)

        modified:   numpyserver/npy_server.hpp

Want to unstage the numpyserver changes as they do not belong in the planned commit.::

    epsilon:opticks blyth$ cp numpyserver/numpydelegate.cpp ~/     ## to be safe 
    epsilon:opticks blyth$ cp numpyserver/numpyserver.hpp ~/

    epsilon:opticks blyth$ git reset HEAD numpyserver/numpydelegate.cpp
    Unstaged changes after reset:
    M	numpyserver/npy_server.hpp
    M	numpyserver/numpydelegate.cpp

    epsilon:opticks blyth$ git reset HEAD numpyserver/numpyserver.hpp
    Unstaged changes after reset:
    M	numpyserver/npy_server.hpp
    M	numpyserver/numpydelegate.cpp
    M	numpyserver/numpyserver.hpp

    epsilon:opticks blyth$ diff ~/numpydelegate.cpp numpyserver/numpydelegate.cpp
    epsilon:opticks blyth$ diff ~/numpyserver.hpp   numpyserver/numpyserver.hpp
    epsilon:opticks blyth$ rm ~/numpydelegate.cpp ~/numpyserver.hpp




git log
---------

::

    git log --stat
    git log --name-status
    git log --name-only


Git LFS
----------

* https://medium.com/swlh/learning-about-git-large-file-system-lfs-72e0c86cfbaf


May 2018 : github permission denied with DSA keys ? RSA still working 
-------------------------------------------------------------------------

Git permission denied over SSH with::

   git clone git@github.com:simoncblyth/bcm.git

Following instructions from https://help.github.com/articles/error-permission-denied-publickey/
Reveals dsa keys no longer working::

   epsilon:~ blyth$ ssh -vT git@github.com
   ...
   debug1: Skipping ssh-dss key /Users/blyth/.ssh/id_dsa - not in PubkeyAcceptedKeyTypes

After adding id_rsa.pub to github webinterface succeed to regain ssh cloning access.


github pull request from a fork
----------------------------------

* https://help.github.com/articles/creating-a-pull-request-from-a-fork/


git : how to list remotes
---------------------------

::

    git remote -v




git syncing a fork
---------------------

* https://help.github.com/articles/syncing-a-fork/

add remote for the upstream repo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    epsilon:plog blyth$ git remote -v
    origin	https://github.com/simoncblyth/plog (fetch)
    origin	https://github.com/simoncblyth/plog (push)
    epsilon:plog blyth$ 
    epsilon:plog blyth$ git remote add upstream https://github.com/SergiusTheBest/plog
    epsilon:plog blyth$ git remote -v
    origin	https://github.com/simoncblyth/plog (fetch)
    origin	https://github.com/simoncblyth/plog (push)
    upstream	https://github.com/SergiusTheBest/plog (fetch)
    upstream	https://github.com/SergiusTheBest/plog (push)
    epsilon:plog blyth$ 


another example of adding upstream remote
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::
     
    (base) epsilon:g4ok blyth$ cd ~/G4OpticksTest
    (base) epsilon:G4OpticksTest blyth$ git remote -v 
    origin	git@github.com:simoncblyth/G4OpticksTest.git (fetch)
    origin	git@github.com:simoncblyth/G4OpticksTest.git (push)
    (base) epsilon:G4OpticksTest blyth$ 
    (base) epsilon:G4OpticksTest blyth$ 
    (base) epsilon:G4OpticksTest blyth$ git remote add upstream https://github.com/hanswenzel/G4OpticksTest
    (base) epsilon:G4OpticksTest blyth$ git remote -v
    origin	git@github.com:simoncblyth/G4OpticksTest.git (fetch)
    origin	git@github.com:simoncblyth/G4OpticksTest.git (push)
    upstream	https://github.com/hanswenzel/G4OpticksTest (fetch)
    upstream	https://github.com/hanswenzel/G4OpticksTest (push)
    (base) epsilon:G4OpticksTest blyth$ 


fetch the changes
~~~~~~~~~~~~~~~~~~~~

::

    epsilon:plog blyth$ git fetch upstream
    remote: Counting objects: 517, done.
    remote: Compressing objects: 100% (28/28), done.
    remote: Total 517 (delta 201), reused 215 (delta 194), pack-reused 288
    Receiving objects: 100% (517/517), 108.88 KiB | 518.00 KiB/s, done.
    Resolving deltas: 100% (312/312), completed with 28 local objects.
    From https://github.com/SergiusTheBest/plog
     * [new branch]      master     -> upstream/master
     * [new tag]         1.1.4      -> 1.1.4
     * [new tag]         1.0.2      -> 1.0.2
     * [new tag]         1.1.0      -> 1.1.0
     * [new tag]         1.1.1      -> 1.1.1
     * [new tag]         1.1.2      -> 1.1.2
     * [new tag]         1.1.3      -> 1.1.3
    epsilon:plog blyth$ 

::

    (base) epsilon:G4OpticksTest blyth$ git fetch upstream
    remote: Enumerating objects: 53, done.
    remote: Counting objects: 100% (53/53), done.
    remote: Compressing objects: 100% (29/29), done.
    remote: Total 53 (delta 34), reused 43 (delta 24), pack-reused 0
    Unpacking objects: 100% (53/53), done.
    From https://github.com/hanswenzel/G4OpticksTest
     * [new branch]      master     -> upstream/master
     * [new tag]         v0.1.1     -> v0.1.1
    (base) epsilon:G4OpticksTest blyth$ 




Merge
~~~~~~~~

::

    epsilon:plog blyth$ git merge upstream/master
    Updating 003f3c6..dcbcca7
    Fast-forward
     .appveyor.yml                                  |  31 ++++++++++++
     ...
     samples/NativeEOL/Main.cpp                     |  21 ++++++++
     samples/Performance/Main.cpp                   |   2 +-
     35 files changed, 1527 insertions(+), 352 deletions(-)
     create mode 100644 .appveyor.yml
     create mode 100644 .circleci/config.yml
     delete mode 100644 appveyor.yml
     create mode 100644 include/plog/Appenders/DebugOutputAppender.h
     ...
     create mode 100644 samples/NativeEOL/Main.cpp
    epsilon:plog blyth$ 



::

    (base) epsilon:G4OpticksTest blyth$ git merge upstream/master
    Removing src/PhysicsList.cc
    Removing src/L4Scintillation.cc
    Removing src/L4Cerenkov.cc
    Removing include/PhysicsList.hh
    Removing include/L4Scintillation.hh
    Removing include/L4Cerenkov.hh
    Merge made by the 'recursive' strategy.
     CMakeLists.txt                    |   2 +-
     G4OpticksTest.cc                  |  20 +++-
     gdml/G4Opticks.gdml               | 142 ++++++++++++-----------
     gdml/G4Opticks_test.gdml          | 193 +++++++++++++++++++++++++++++++
     include/G4.hh                     |   4 +-
     include/L4Cerenkov.hh             | 263 ------------------------------------------
     include/L4Scintillation.hh        | 406 -----------------------------------------------------------------
     include/PhysicsList.hh            |  53 ---------
     include/PrimaryGeneratorAction.hh |  71 ++++++------
     include/lArTPCSD.hh               |  52 +++++++++
     src/DetectorConstruction.cc       |  56 +++------
     src/EventAction.cc                |   3 +-
     src/G4.cc                         | 103 ++++++++++++++++-
     src/L4Cerenkov.cc                 | 859 -----------------------------------------------------------------------------------------------------------------------------------------
     src/L4Scintillation.cc            | 962 ----------------------------------------------------------------------------------------------------------------------------------------------------------
     src/PhysicsList.cc                | 227 -------------------------------------
     src/PrimaryGeneratorAction.cc     |  96 ++++++++--------
     src/SensitiveDetector.cc          |   5 +-
     src/TrackerSD.cc                  |  22 +---
     src/lArTPCSD.cc                   | 167 +++++++++++++++++++++++++++
     20 files changed, 712 insertions(+), 2994 deletions(-)
     create mode 100644 gdml/G4Opticks_test.gdml
     delete mode 100644 include/L4Cerenkov.hh
     delete mode 100644 include/L4Scintillation.hh
     delete mode 100644 include/PhysicsList.hh
     create mode 100644 include/lArTPCSD.hh
     delete mode 100644 src/L4Cerenkov.cc
     delete mode 100644 src/L4Scintillation.cc
     delete mode 100644 src/PhysicsList.cc
     create mode 100644 src/lArTPCSD.cc
    (base) epsilon:G4OpticksTest blyth$ 

    (base) epsilon:G4OpticksTest blyth$ git status 
    On branch master
    Your branch is ahead of 'origin/master' by 10 commits.
      (use "git push" to publish your local commits)

    nothing to commit, working tree clean
    (base) epsilon:G4OpticksTest blyth$ 
    (base) epsilon:G4OpticksTest blyth$ 
    (base) epsilon:G4OpticksTest blyth$ git push 
    Counting objects: 1, done.
    Writing objects: 100% (1/1), 245 bytes | 245.00 KiB/s, done.
    Total 1 (delta 0), reused 0 (delta 0)
    To github.com:simoncblyth/G4OpticksTest.git
       437a9fc..277faa7  master -> master
    (base) epsilon:G4OpticksTest blyth$ git status
    On branch master
    Your branch is up-to-date with 'origin/master'.

    nothing to commit, working tree clean
    (base) epsilon:G4OpticksTest blyth$ 








git ignores with Xcode 
------------------------

* https://www.raywenderlich.com/153084/use-git-source-control-xcode-9
* https://git-scm.com/book/en/v2/Git-Basics-Git-Aliases

::

    git config --global alias.ignore '!gi() { curl -L -s https://www.gitignore.io/api/$@ ;}; gi'
    git ignore swift,macos >.gitignore

Global aliases are stored in ~/.gitconfig 


Git Books
-----------

* https://git-scm.com/book/en/v2

Getting Git on Server
-----------------------

* https://git-scm.com/book/en/v2/Git-on-the-Server-Getting-Git-on-a-Server#_getting_git_on_a_server


Fixing ssh environment to find newer than system git
-----------------------------------------------------

Cloning a bare repo over ssh from g4pb initially fails for lack 
of git-upload-pack in sshd PATH environment.
Succeeds after setup environment of sshd in g4pb (see dsmgit-).

Handling pull Aborted
------------------------

::

    delta:home blyth$ git pull
    remote: Counting objects: 13, done.
    remote: Compressing objects: 100% (7/7), done.
    remote: Total 7 (delta 6), reused 0 (delta 0)
    Unpacking objects: 100% (7/7), done.
    From file:///Volumes/UOWStick/var/scm/git/home
       da974ca..4205a75  master     -> uow/master
    Updating da974ca..4205a75
    error: Your local changes to the following files would be overwritten by merge:
        sysadmin/defaults.bash
        sysadmin/migration.bash
    Please, commit your changes or stash them before you can merge.
    Aborting
    delta:home blyth$ 


Comparisons suggest identical changes already there, but not committed::

    git show uow/master:sysadmin/defaults.bash
    git diff uow/master:sysadmin/defaults.bash sysadmin/defaults.bash

Hmm this arises from using one macOS but 
changing files from the older macOS volume.
So drop the changes, but make a safety copy to /tmp::

    delta:home blyth$ cp sysadmin/migration.bash /tmp/
    delta:home blyth$ cp sysadmin/defaults.bash /tmp/

    git checkout -- sysadmin/defaults.bash
    git checkout -- sysadmin/migration.bash

Now the pulling completes::

    delta:home blyth$ git pull
    Updating da974ca..4205a75
    Fast-forward
     sysadmin/defaults.bash    |   3 +++
     sysadmin/diskutility.bash |  63 +++++++++++++++++++++++++++++++++++++++++++++++
     sysadmin/migration.bash   |  51 +++++++++++++++++++++++++++++++++++++-
     sysadmin/upgrade.bash     | 189 ++++++++++++++++++++++++++++++++++++++++++--------------------------------------------------------------------------------------------------
     4 files changed, 172 insertions(+), 134 deletions(-)
    delta:home blyth$ 




Fetch a remote repo into an existing directory with bystanders
----------------------------------------------------------------

* avoid accidents by setting .gitignore to "*" so must always force to add things to repo

::

    cd /to/the/dir/
    git init                      ## creates .git dir
    git remote add <rnam> <url>    ## configure the remote
    git fetch <rnam> master        ## fetch objects from remote, without yet changing working copy

    git diff <rnam>/master                 ## compare current with what about to checkout
    git show <rnam>/master:.workflow.cnf   ## take a closer look

    git checkout master           ## now update working copy 
    ## hmm merge : shouldnt be needed, only one line ...


* note that this doesnt use "clone" as that creates a new dir, fetches and merges 



fetch into non empty dir
--------------------------

* https://stackoverflow.com/questions/2411031/how-do-i-clone-into-a-non-empty-directory

::

    git init

    git remote add origin PATH/TO/REPO

    git fetch

    git reset origin/master         
        # copies entries from origin/master  to the index
        # this is required if files in the non-empty directory are in the repo

    git checkout -t origin/master   # -t for --track
        # from index to working tree


git help reset
----------------

::

   git reset [-q] [<tree-ish>] [--] <paths>...

       This form resets the index entries for all <paths> to their state at
       <tree-ish>. (It does not affect the working tree, nor the current branch.)

       This means that git reset <paths> is the opposite of git add <paths>.

       After running git reset <paths> to update the index entry, you can use
       git-checkout(1) to check the contents out of the index to the working tree.
       Alternatively, using git-checkout(1) and specifying a commit, you can copy the
       contents of a path out of a commit to the index and to the working tree in one go.


So *reset* used like this enables a more controlled way of 
updating the working tree.  First from treeish into index and then checkout 
from index into working tree.




.gitignore vs .git/info/exclude
----------------------------------

::

   man gitignore


.gitignore 
    when committed it is shared by all clones of the repo 

.git/info/exclude 
    only applies to the local copy of the repository, not versioned


Patterns which a user wants Git to ignore in all situations 
(e.g., backup or temporary files generated by the user's editor of choice) generally go into a file
specified by core.excludesfile in the user's ~/.gitconfig. Its default value is $XDG_CONFIG_HOME/git/ignore. 
If $XDG_CONFIG_HOME is either not set or empty,
$HOME/.config/git/ignore is used instead.



Showing old versions of a file
---------------------------------

::

    delta:bin blyth$ git lg dot.py 
    * 1a20d0b - (uow/master, g4pb/master, arc/master) try direct HOME/.git approach to dot-backup with file/dir permission fixing (15 hours ago) <Simon C Blyth>
    * 6c48db5 - dot-backup dot-recover machinery using USB stick remote (2 days ago) <Simon C Blyth>
    * 5f5861b - dot machinery working to some extent, need to test permission recovery from fresh clones of dot repo (3 days ago) <Simon C Blyth>
    * 4a07bea - start dot backup/recover machinery (3 days ago) <Simon C Blyth>
    delta:bin blyth$ git show 4a07bea:bin/dot.py 



Git And Filemodes
--------------------

* https://stackoverflow.com/questions/4832346/how-to-use-group-file-permissions-correctly-on-a-git-repository

::

    git archive --format=tar --prefix=junk/ HEAD | (cd /var/tmp/ && tar xf -)



Git just seems to distinguish executable or not, other modes not preserved::

    delta:test blyth$ git ls-files --stage
    100644 e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 0   a.txt
    100644 e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 0   b.txt
    100755 e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 0   c.txt
    100755 e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 0   d.txt
    delta:test blyth$ l *.txt
    -rwxrwxrwx  1 blyth  wheel  0 Mar 15 13:24 d.txt
    -rwxr-xr-x  1 blyth  wheel  0 Mar 15 13:23 c.txt
    -rw-------  1 blyth  wheel  0 Mar 15 13:22 b.txt
    -rw-r--r--  1 blyth  wheel  0 Mar 15 13:21 a.txt
    delta:test blyth$ 


* https://git-scm.com/docs/git-archive
* https://git-scm.com/docs/git-archive#ATTRIBUTES

* https://github.com/dr4Ke/git-preserve-permissions/blob/master/git-preserve-permissions

  overcomplicated perl approach  






list authors of a git repository including commit count and email
-------------------------------------------------------------------

::

   git shortlog -e -s -n
   git shortlog -esn


-n, --numbered
   Sort output according to the number of commits per author instead of author alphabetic order.

-s, --summary
   Suppress commit description and provide a commit count summary only.

-e, --email
   Show the email address of each author.




Git and deployment
--------------------


* https://github.com/git-deploy/git-deploy

  over the top approach

* http://gitolite.com/deploy.html

  run down of different approaches


git archive --format=tar --prefix=junk/ HEAD | (cd /var/tmp/ && tar xf -)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://feeding.cloud.geek.nz/posts/excluding-files-from-git-archive/

* https://github.com/git/git/blob/master/builtin/archive.c



Git tags : lightweight or annotated
--------------------------------------

* https://git-scm.com/book/en/v2/Git-Basics-Tagging

A lightweight tag is very much like a branch that doesn’t change — it’s just a
pointer to a specific commit.

Annotated tags, however, are stored as full objects in the Git database.
They’re checksummed; contain the tagger name, email, and date; have a tagging
message; and can be signed and verified with GNU Privacy Guard (GPG). It’s
generally recommended that you create annotated tags so you can have all this
information; but if you want a temporary tag or for some reason don’t want to
keep the other information, lightweight tags are available too.


* https://www.atlassian.com/git/tutorials/inspecting-a-repository/git-tag

A best practice is to consider Annotated tags as public, and Lightweight tags as private.

Lightweight tags are essentially 'bookmarks' to a commit, they are just a name
and a pointer to a commit, useful for creating quick links to relevant commits.


::

   git tag v1.4-lw   # lite 
   git tag  # list them



Making the first tag
~~~~~~~~~~~~~~~~~~~~~~~

::

    epsilon:opticks blyth$ git tag -a v0.0.0-rc2 -m "2nd test tag"
    epsilon:opticks blyth$ git tag -a v0.0.0-rc3 -m "3rd test tag"
    epsilon:opticks blyth$ git tag
    v0.0.0-rc1

    epsilon:opticks blyth$ git show v0.0.0-rc1
    tag v0.0.0-rc1
    Tagger: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Fri Jul 3 20:25:04 2020 +0100

    first test release

    commit c008637d663cad752337523312e888362fd4df90 (HEAD -> master, tag: v0.0.0-rc1, origin/master, origin/HEAD, github/master)
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Fri Jul 3 20:19:36 2020 +0100

        update README with git urls and the new github opticks repo



3rd tag
~~~~~~~~~~~

::

    epsilon:opticks blyth$ git tag -a v0.0.0-rc3 -m "3rd test tag"
    epsilon:opticks blyth$ git push --tags
    epsilon:opticks blyth$ git push github --tags

    epsilon:opticks blyth$ git push github   # done this before : pushing the tags before the commits, seems not to cause an issue




Delete the first tag
~~~~~~~~~~~~~~~~~~~~~~~~

Local delete::

    epsilon:opticks blyth$ git tag -d v0.0.0-rc1
    Deleted tag 'v0.0.0-rc1' (was 5658448b)


* github has web interface for deleting the remote tag 
* bitbucket : dont see any web interface

::
   
    epsilon:opticks blyth$ git push origin :refs/tags/v0.0.0-rc1  
    To bitbucket.org:simoncblyth/opticks.git
     - [deleted]           v0.0.0-rc1



Making and pushing 2nd tag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    epsilon:opticks blyth$ git tag -a v0.0.0-rc2 -m "2nd test tag"
    epsilon:opticks blyth$ git push --tags
    epsilon:opticks blyth$ git push github --tags


customgeant4 first tag
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    c4 

    git tag -a v0.0.1 -m "initial tag"
    git push --tags

    curl -L -O https://github.com/simoncblyth/customgeant4/archive/refs/tags/v0.0.1.zip


    epsilon:tmp blyth$ unzip v0.0.1.zip
    Archive:  v0.0.1.zip
    6c3f976abc213228307dd4d6237771bac2800458
       creating: customgeant4-0.0.1/
      inflating: customgeant4-0.0.1/.gitignore  
      inflating: customgeant4-0.0.1/C4CustomART.h  
      inflating: customgeant4-0.0.1/C4CustomART_Debug.h  
      inflating: customgeant4-0.0.1/C4IPMTAccessor.h  
      inflating: customgeant4-0.0.1/C4MultiLayrStack.h  





Tags on Bitbucket and Github web interface
---------------------------------------------

* https://bitbucket.org/simoncblyth/opticks/src/v0.0.0-rc1/
* https://github.com/simoncblyth/opticks/releases/tag/v0.0.0-rc1   
* https://github.com/simoncblyth/opticks/archive/v0.0.0-rc1.tar.gz



Checkout tags
---------------

::

   git checkout v1.4

This puts the repo in a detached HEAD state. This means any changes made will
not update the tag. They will create a new detached commit. This new detached
commit will not be part of any branch and will only be reachable directly by
the commits SHA hash. Therefore it is a best practice to create a new branch
anytime you're making changes in a detached HEAD state.



Sharing Tags
---------------

By default, the git push command doesn’t transfer tags to remote servers. You
will have to explicitly push tags to a shared server after you have created
them. This process is just like sharing remote branches — you can run git push
origin <tagname>.

::

    git push origin v1.5
    git push origin --tags


Tagging Releases
--------------------

* https://dev.to/neshaz/a-tutorial-for-tagging-releases-in-git-147e






extending git with custom commands
--------------------------------------

* https://www.atlassian.com/blog/git/extending-git

::

    delta:home blyth$ git demo one two three
    ['/Users/blyth/env/bin/git-demo', 'one', 'two', 'three']

    delta:home blyth$ cat ~/env/bin/git-demo 
    #!/usr/bin/env python
    import sys
    print sys.argv




Log Alias
------------

* https://coderwall.com/p/euwpig/a-better-git-log

::

    git log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit

    git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"

    git config --global alias.lga "log --graph --pretty=format:'%h -%d %s (%cr) <%an>' --abbrev-commit"


::

    git lg 
    git lg -p



Remove Old history from git repo ?
-------------------------------------

Via replace
~~~~~~~~~~~~~

* https://git-scm.com/book/en/v2/Git-Tools-Replace

Via graft
~~~~~~~~~~~

* https://stackoverflow.com/questions/4515580/how-do-i-remove-the-old-history-from-a-git-repository

* https://git.wiki.kernel.org/index.php/GraftPoint

.git/info/grafts has two sha1 separated by space and terminated by newline.

from `git help branch-rewrite`::

    NOTE: This command honors .git/info/grafts file and refs in the refs/replace/
    namespace. If you have any grafts or replacement refs defined, running this
    command will make them permanent.
     


Git Basics
-----------

* https://git-scm.com/book/en/v2/Getting-Started-Git-Basics

SVN, Hg are delta-based version control.
Git conceptually is snapshot-based (not delta).


`git status -s/--short`
~~~~~~~~~~~~~~~~~~~~~~~~~~

Two column output 

- left column indicates status of staging area 
- right column indicates status of working tree


`git diff`
~~~~~~~~~~~~

Compare working tree with staging area (aka index).


`git diff --staged/--cached`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare staging area with last commit.
 



Working with multiple remotes
------------------------------

* https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes
* https://git-scm.com/book/id/v2/Git-Branching-Remote-Branches

git pull
~~~~~~~~~~

If your current branch is set up to track a remote branch (see the next section
and Git Branching for more information), you can use the git pull command to
automatically fetch and then merge that remote branch into your current branch.


`git push -u/--set-upstream`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For every branch that is up to date or successfully pushed, add upstream
(tracking) reference, used by argument-less git-pull(1) and other commands. For
more information, see branch.<name>.merge in git-config(1).

Note only one remote can be the default upstream, for that one 
the argument-less::

    git push -v   
    git pull -v

is equivalent to::

    git push g4pb master -v
    git pull g4pb master -v

For other remotes need to use full form, eg::

    git push arc master -v     
    git pull arc master -v


`git show-branch *master`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://git-scm.com/book/id/v2/Git-Branching-Remote-Branches

::

    delta:home blyth$ git show-branch *master
    * [master] in all bash functions switch the informational workflow-home to home-home
     ! [refs/remotes/arc/master] in all bash functions switch the informational workflow-home to home-home
      ! [refs/remotes/g4pb/master] in all bash functions switch the informational workflow-home to home-home
    ---
    *++ [master] in all bash functions switch the informational workflow-home to home-home
    delta:home blyth$ 
        

`git rev-parse master g4pb/master arc/master`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    delta:home blyth$ git rev-parse master g4pb/master arc/master 
    5f5247608802bebd17a7952167d3cbd24a6912bd
    5f5247608802bebd17a7952167d3cbd24a6912bd
    5f5247608802bebd17a7952167d3cbd24a6912bd



Take a look at Sphinx tags/1.2, then return to latest, see sphinx-
---------------------------------------------------------------------------

::

    delta:sphinx blyth$ git checkout tags/1.2
    Note: checking out 'tags/1.2'.

    You are in 'detached HEAD' state. You can look around, make experimental
    changes and commit them, and you can discard any commits you make in this
    state without impacting any branches by performing another checkout.

    If you want to create a new branch to retain commits you create, you may
    do so (now or later) by using -b with the checkout command again. Example:

      git checkout -b new_branch_name

    HEAD is now at 2a86eff... Changelog bump.
    delta:sphinx blyth$ 

    delta:sphinx blyth$ git status
    HEAD detached at 1.2
    nothing to commit, working directory clean

    delta:sphinx blyth$ git checkout master
    Previous HEAD position was 2a86eff... Changelog bump.
    Switched to branch 'master'
    Your branch is up-to-date with 'origin/master'.

    delta:sphinx blyth$ git status
    On branch master
    Your branch is up-to-date with 'origin/master'.

    nothing to commit, working directory clean



Github Merging a pull request via web interface
--------------------------------------------------

* click on title of the pull request (not the selection button)
* use the dropdown to right to pick the type of merge then click the button to left
* add commit message and confirm the merge


Status
--------

::

    # porcelain: line-by-line format for scripts , branch: show the branch and tracking info
    delta:assimp-fork blyth$ git status --porcelain --branch   
    ## master...origin/master
     M Readme.md
    ?? hello.txt


Config
--------

::

    delta:~ blyth$ git config --list
    user.name=Simon C Blyth
    user.email=simon.c.blyth@gmail.com
    color.diff=auto
    color.status=auto
    color.branch=auto
    core.repositoryformatversion=0
    core.filemode=true
    core.bare=false
    core.logallrefupdates=true
    core.ignorecase=true
    core.precomposeunicode=true
    remote.origin.url=git@bitbucket.org:simoncblyth/testhome.git
    remote.origin.fetch=+refs/heads/*:refs/remotes/origin/*
    branch.master.remote=origin
    branch.master.merge=refs/heads/master
    delta:~ blyth$ 

    delta:hometest blyth$ git config --list
    user.name=Simon C Blyth
    user.email=simon.c.blyth@gmail.com
    color.diff=auto
    color.status=auto
    color.branch=auto
    core.repositoryformatversion=0
    core.filemode=true
    core.bare=false
    core.logallrefupdates=true
    core.ignorecase=true
    core.precomposeunicode=true
    push.default=simple
    remote.origin.url=gcrypt::blyth@192.168.0.200:testhomecrypt.git
    remote.origin.fetch=+refs/heads/*:refs/remotes/origin/*
    remote.origin.gcrypt-id=:id:poByIm5T1tjn3sXlDiVg
    delta:hometest blyth$ 





Git Remote Helpers
--------------------

* https://git-scm.com/docs/git-remote-helpers

Remote helper programs are normally not used directly by end users, but they
are invoked by git when it needs to interact with remote repositories git does
not support natively. A given helper will implement a subset of the
capabilities documented here. When git needs to interact with a repository
using a remote helper, it spawns the helper as an independent process, sends
commands to the helper’s standard input, and expects results from the helper’s
standard output. Because a remote helper runs as an independent process from
git, there is no need to re-link git to add a new helper, nor any need to link
the helper with the implementation of git.

When git encounters a URL of the form <transport>://<address>, where
<transport> is a protocol that it cannot handle natively, it automatically
invokes git remote-<transport> with the full URL as the second argument.


Encrypted Git Repo ?
---------------------

* see w-/gcrypt-


git remote show
----------------

::

    delta:~ blyth$ git remote -v show
    origin  git@bitbucket.org:simoncblyth/testhome.git (fetch)
    origin  git@bitbucket.org:simoncblyth/testhome.git (push)


Avoid http blockages by cloning over SSH 
-------------------------------------------

* Sometimes can avoid "http" blockage by using "https" to use a different port 


* https://stackoverflow.com/questions/6167905/git-clone-through-ssh

Git URL in one of two forms:

ssh://username@host.xz/absolute/path/to/repo.git/ 
    just a forward slash for absolute path on server

username@host.xz:relative/path/to/repo.git/ 
    just a colon (it mustn't have the ssh:// for relative path on server (relative to home dir of username on server machine)


Note this did not work with shallow (--depth 1) clones, possibly as git version
on one of the machines is too old.
 

clone into home on unblocked machine 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
::

    simon:~ blyth$ git clone http://git.videolan.org/git/x264
    Cloning into 'x264'...
    remote: Counting objects: 20569, done.
    remote: Compressing objects: 100% (4266/4266), done.
    remote: Total 20569 (delta 17005), reused 19712 (delta 16260)
    Receiving objects: 100% (20569/20569), 4.83 MiB | 76.00 KiB/s, done.
    Resolving deltas: 100% (17005/17005), done.
    Checking connectivity... done.
    simon:~ blyth$ 


clone over ssh from http blocked machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [simon@localhost x264]$ git clone blyth@simon.phys.ntu.edu.tw:x264
    Initialized empty Git repository in /usr/local/env/video/x264/x264/.git/
    Password:
    remote: Counting objects: 20569, done.
    remote: Compressing objects: 100% (3521/3521), done.
    remote: Total 20569 (delta 17005), reused 20569 (delta 17005)
    Receiving objects: 100% (20569/20569), 4.83 MiB | 547 KiB/s, done.
    Resolving deltas: 100% (17005/17005), done.
    [simon@localhost x264]$ 


git remote add
---------------

* https://caolan.org/posts/encrypted_git_repositories.html

::

    git remote add origin gcrypt::rsync://username@your-server.com:example-crypt
    git remote add origin gcrypt::rsync://username@your-server.com:example-crypt


git push : setting push.default
-----------------------------------

::

    delta:hometest blyth$ git push 
    warning: push.default is unset; its implicit value is changing in
    Git 2.0 from 'matching' to 'simple'. To squelch this message
    and maintain the current behavior after the default changes, use:

      git config --global push.default matching

    To squelch this message and adopt the new behavior now, use:

      git config --global push.default simple

    When push.default is set to 'matching', git will push local branches
    to the remote branches that already exist with the same name.

    In Git 2.0, Git will default to the more conservative 'simple'
    behavior, which only pushes the current branch to the corresponding
    remote branch that 'git pull' uses to update the current branch.

    See 'git help config' and search for 'push.default' for further information.
    (the 'simple' mode was introduced in Git 1.7.11. Use the similar mode
    'current' instead of 'simple' if you sometimes use older versions of Git)

    blyth@192.168.0.200's password: 
    Everything up-to-date

::

    delta:hometest blyth$ git config push.default simple
    delta:hometest blyth$ git push 
    blyth@192.168.0.200's password: 
    Everything up-to-date

After adding key to .ssh/authorized_keys2 on server, and starting ssh agent : works without password::

    delta:hometest blyth$ git push 
    Everything up-to-date



git pull
-----------

* combined fetch and merge


Git-hg-rosetta-stone
--------------------------

* https://github.com/sympy/sympy/wiki/Git-hg-rosetta-stone

Main difference is the git *index*, a staging area for preparing commits


================================  =======================================  ===========================================
hg                                 git
================================  =======================================  ===========================================
hg revert some_file                git checkout some_file
--------------------------------  ---------------------------------------  -------------------------------------------
                                   git help checkout                        updates the named paths in the working tree from the index file or from a
                                                                            named <tree-ish> (most often a commit)  
--------------------------------  ---------------------------------------  -------------------------------------------
hg backout                         git revert
--------------------------------  ---------------------------------------  -------------------------------------------
hg add                             git add                                  git adds to index
================================  =======================================  ===========================================



Whats all this ?
-----------------

::

    simon:boost-python-examples blyth$ git add 04-ClassMembers/member.py
    simon:boost-python-examples blyth$ git commit -m "avoid uninteresting error regards non-ascii encodings"
    [master 1abe2be] avoid uninteresting error regards non-ascii encodings
     1 file changed, 1 insertion(+), 1 deletion(-)

    simon:boost-python-examples blyth$ git push 
    warning: push.default is unset; its implicit value is changing in
    Git 2.0 from 'matching' to 'simple'. To squelch this message
    and maintain the current behavior after the default changes, use:

      git config --global push.default matching

    To squelch this message and adopt the new behavior now, use:

      git config --global push.default simple

    When push.default is set to 'matching', git will push local branches
    to the remote branches that already exist with the same name.

    In Git 2.0, Git will default to the more conservative 'simple'
    behavior, which only pushes the current branch to the corresponding
    remote branch that 'git pull' uses to update the current branch.

    See 'git help config' and search for 'push.default' for further information.
    (the 'simple' mode was introduced in Git 1.7.11. Use the similar mode
    'current' instead of 'simple' if you sometimes use older versions of Git)

    Username for 'https://github.com': simoncblyth
    Password for 'https://simoncblyth@github.com': 
    Counting objects: 7, done.
    Delta compression using up to 8 threads.
    Compressing objects: 100% (4/4), done.
    Writing objects: 100% (4/4), 376 bytes | 0 bytes/s, done.
    Total 4 (delta 3), reused 0 (delta 0)
    To https://github.com/simoncblyth/boost-python-examples
       90f6092..1abe2be  master -> master
    simon:boost-python-examples blyth$ 
    simon:boost-python-examples blyth$ 




Git workflow
---------------

Make mods

* "git status" 
* "git add" changed files
* "git commit -m  




Failed push, seems must now use https
---------------------------------------

::

    delta:DualContouringSample blyth$ git push 
    warning: push.default is unset; its implicit value is changing in
    Git 2.0 from 'matching' to 'simple'. To squelch this message
    and maintain the current behavior after the default changes, use:

      git config --global push.default matching

    To squelch this message and adopt the new behavior now, use:

      git config --global push.default simple

    When push.default is set to 'matching', git will push local branches
    to the remote branches that already exist with the same name.

    In Git 2.0, Git will default to the more conservative 'simple'
    behavior, which only pushes the current branch to the corresponding
    remote branch that 'git pull' uses to update the current branch.

    See 'git help config' and search for 'push.default' for further information.
    (the 'simple' mode was introduced in Git 1.7.11. Use the similar mode
    'current' instead of 'simple' if you sometimes use older versions of Git)

    fatal: remote error: 
      You can't push to git://github.com/simoncblyth/DualContouringSample.git
      Use https://github.com/simoncblyth/DualContouringSample.git
    delta:DualContouringSample blyth$ 




Updating from remote branch
----------------------------

  git pull origin master    # 

Simple branching
------------------

* http://git-scm.com/book/en/Git-Branching-Basic-Branching-and-Merging

::

    git checkout -b py25compat    # shorthand for "git branch" "git checkout" of the named branch  


Sharing a git repo
---------------------

* http://www.git-scm.com/book/en/Git-on-the-Server-The-Protocols#The-HTTP/S-Protocol

Whats all this ?
----------------

delta:code blyth$ git push 
warning: push.default is unset; its implicit value is changing in
Git 2.0 from 'matching' to 'simple'. To squelch this message
and maintain the current behavior after the default changes, use:

  git config --global push.default matching

To squelch this message and adopt the new behavior now, use:

  git config --global push.default simple

When push.default is set to 'matching', git will push local branches
to the remote branches that already exist with the same name.

In Git 2.0, Git will default to the more conservative 'simple'
behavior, which only pushes the current branch to the corresponding
remote branch that 'git pull' uses to update the current branch.

See 'git help config' and search for 'push.default' for further information.
(the 'simple' mode was introduced in Git 1.7.11. Use the similar mode
'current' instead of 'simple' if you sometimes use older versions of Git)

Counting objects: 49, done.
Delta compression using up to 8 threads.
Compressing objects: 100% (8/8), done.
Writing objects: 100% (8/8), 3.40 KiB | 0 bytes/s, done.
Total 8 (delta 7), reused 0 (delta 0)
To git@github.com:simoncblyth/assimp.git
   845e88a..1ff18aa  master -> master
delta:code blyth$ 




FUNCTIONS
----------

*git-bare*
       when invoked from the root of git working copy this
       creates a bare repo in *git-bare-dir* eg /var/scm/git 

*git-bare-scp name tag* 
       scp the bare git repo to remote node  


EOU
}
git-dir(){ echo $(local-scm-fold)/git ; }
git-cd(){  cd $(git-dir); }
git-mate(){ mate $(git-dir) ; }
git-make(){
   local dir=$(git-dir) &&  mkdir -p $dir 
}

git-bare-dir(){ echo $(git-dir) ; }
git-bare-path(){ echo $(git-bare-dir)/${1:-dummy}.git ; }
git-bare-name(){ echo ${GIT_BARE_NAME:-pycollada} ;}

git-bare(){
  local msg="=== $FUNCNAME :"
  echo $msg following recipe from http://www.git-scm.com/book/en/Git-on-the-Server-The-Protocols#The-HTTP/S-Protocol
  local path=$PWD
  [ ! -d "${path}/.git" ] && echo $msg needs to be invoked from toplevel of git checkout containing .git folder   && return 1
  local name=$(basename $path)
  local bare=$(git-bare-path $name)
  local hook=$bare/hooks/post-update
  [ -d "$bare" ] && echo $msg bare repo exists already at $bare && return 1
  local cmd="git clone --bare $path $bare ; mv $hook.sample $hook ; chmod a+x $hook  "
  echo $msg $cmd
  eval $cmd
}

git-bare-scp(){
  local msg="=== $FUNCNAME :"
  local name=${1:-$(git-bare-name)}
  local tag=${2:-N}
  [ "$NODE_TAG" == "$tag" ] && echo $msg cannot scp to self $tag && return 1 
  local cmd="scp -r $(git-bare-path $name) $tag:$(NODE_TAG=$tag git-bare-dir)"
  echo $msg $cmd
  eval $cmd
}

git-bare-clone(){
  local msg="=== $FUNCNAME :"
  local name=${1:-$(git-bare-name)}
  local bare=$(git-bare-path $name)
  [ ! -d "$bare" ] && echo $msg no bare git repo at $bare && return 1
  local cmd="git clone $bare"
  echo $msg $cmd
  eval $cmd 
}


git-origin(){
   git remote show origin
}


git-edit(){ vi ~/.gitconfig ; }

git-conf(){

git config --global user.name "Simon C Blyth"
git config --global user.email "simon.c.blyth@gmail.com"
git config --global color.diff auto
git config --global color.status auto
git config --global color.branch auto
#git config --global core.editor "mate -w"

git config -l

}

git-learning(){

  local dir=/tmp/env/$FUNCNAME && mkdir -p $dir
  cd $dir


  git clone git://github.com/dcramer/django-compositepks.git
  git clone git://github.com/django/django.git


}


git-global-ignores-(){ cat << EOI
# $FUNCNAME
.DS_Store
*.swp
*.pyc
EOI
}

git-global-ignores(){
   local path=$HOME/.config/git/ignore 

   [ -f $path ] && echo $msg path $path exists already && return 

   local dir=$(dirname $path)
   mkdir -p $dir 

   $FUNCNAME- 
   echo $msg writing above to path $path : see man gitignore   

   $FUNCNAME- > $path
}



git-upstream-notes(){ cat << EON

https://stackoverflow.com/questions/4950725/how-can-i-see-which-git-branches-are-tracking-which-remote-upstream-branch

EON
}

git-upstream(){
   # find the upstream of master branch 
   git rev-parse --abbrev-ref master@{upstream}
}


git-year(){ echo ${GIT_YEAR:-$(date +"%Y")} ; }

git-month-notes(){ cat << EON
Negate the month argument for last year.
Adapted from hg-month

Second argument controls the format of the logging, see ~/.gitconfig
for possibilities.

lg : commit message one line 
l  : longer with paths changed
ls : cute ascii showing how much changed 

Eg::

    cd ~/opticks
    git-month 8 ls   ## August commits  

EON
}

git-month(){ 
   local arg=${1:-8}
   local log=${2:-lg}
   local year=$(git-year) 

   [ "${arg:0:1}" == "-" ] && arg=${arg:1} && year=$(( $year - 1))

   local beg=$(printf "%0.2u" $arg)
   local end=$(printf "%0.2u" $(( $arg + 1)))

   local byear=$year
   local eyear=$year

   [ "${end}" == "13" ] && end="01" && eyear=$(( $byear + 1  ))

   local cmd="git $log --after $byear-$beg-01 --before $eyear-$end-01 "

   local reverse=1
   if [ $reverse -eq 1 ]; then 
       case $(uname) in
          Darwin) cmd="$cmd | tail -r" ;;
          Linux)  cmd="$cmd | tac" ;;
       esac
   fi 

   echo $cmd
   eval $cmd
}


git-export()
{
    local outdir=$1
    local msg="=== $FUNCNAME :"   
    local name=$(basename $PWD)

    local tmpdir=/tmp/$USER/git-export
    mkdir -p $tmpdir
    echo $msg name $name into $tmpdir/$name

    git archive --format=tar --prefix=$name/ HEAD | (cd $tmpdir && tar xf -)
}



git-socks(){ socks- ;  git config --global http.proxy "socks5://127.0.0.1:$(socks-port)" ; cat $(git-config) ;  }
git-socks-unset(){     git config --global --unset http.proxy  ; cat $(git-config) ; }
git-config(){          echo $HOME/.gitconfig ; }
git-e(){               vi $(git-config) ; }


git-socks-notes(){ cat << EON

Adds the below to ~/.gitconfig::

    [http]
        proxy = socks5://127.0.0.1:8080

EON
}






git-10(){  type $FUNCNAME ; git diff @~10 --name-status ; }


git-noeol-0(){ 
   git ls-files -z | xargs -0 -L1 bash -c 'test "$(tail -c 1 "$0")" && echo $0'
   : env/git/git.bash
} 

git-ls-files-excluding-root-png(){
   git ls-files -- . ':!:*.png' . ':!:*.root'
   : env/git/git.bash 
}


git-noeol(){
   git ls-files -z -- . ':!:*.png' . ':!:*.root'  | xargs -0 -L1 bash -c 'test "$(tail -c 1 "$0")" && echo $0'
   : list files in git repo excluding .png and .root which do not have an eol 
   : env/git/git.bash 
}

git-noeol-fix(){
   : env/git/git.bash 
   : fix files in git repo excluding .png and .root which do not have an eol 

   echo $FUNCNAME : git ls-files search excluding .root .png for files without eol beneath PWD $PWD  
   local paths=$(git-noeol)

   echo $FUNCNAME : found
   for path in $paths ; do printf "$path\n" ; done
 
   echo $FUNCNAME : fixing them with an echo 
   for path in $paths ; do echo >> "$path" ; done 
}





