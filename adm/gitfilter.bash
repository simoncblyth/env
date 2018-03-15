# === func-gen- : adm/gitfilter fgp adm/gitfilter.bash fgn gitfilter fgh adm
gitfilter-src(){      echo adm/gitfilter.bash ; }
gitfilter-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gitfilter-src)} ; }
gitfilter-vi(){       vi $(gitfilter-source) ; }
gitfilter-env(){      elocal- ; }
gitfilter-usage(){ cat << EOU

gitfilter : using filter-branch subcommand
===========================================

overview
----------

* extract some folders from a git repo into a new one
* chop the history 


procedure for workflow_home was automated
-------------------------------------------

* check gitfilter-repo is *workflow_home*
* run the partition:

::

   gitfilter-;gitfilter--

* say YES to cloning into HOME


procedure for workflow_workflow is semi-automated
---------------------------------------------------

* check gitfilter-repo is *workflow_workflow*
* run the partition::

     gitfilter-;gitfilter--

* say N to cloning into HOME, as need to chop first

* examine the logs see gitfilter-chop-notes to pick the 
  SHA1 of the last commit to chop, set gitfilter-chop-last

* run the chop::

     gitfilter-;gitfilter-chop

* do the clone into home::

     gitfilter-;gitfilter-partition-clone
 

Background refs
-----------------

::

     git help filter-branch

* https://confluence.atlassian.com/bitbucket/split-a-repository-in-two-313464964.html
* https://help.github.com/articles/splitting-a-subfolder-out-into-a-new-repository/
* https://stackoverflow.com/questions/359424/detach-move-subdirectory-into-separate-git-repository/17864475#17864475
* https://stackoverflow.com/questions/359424/detach-move-subdirectory-into-separate-git-repository
* https://stackoverflow.com/questions/2982055/detach-many-subdirectories-into-a-new-separate-git-repository
* https://www.atlassian.com/blog/git/tear-apart-repository-git-way

Rewriting means are changing entrire history so the connection with origin
is broken::

    delta:workflow blyth$ git remote -v
    origin  file:///usr/local/env/adm/svn2git/workflow (fetch)
    origin  file:///usr/local/env/adm/svn2git/workflow (push)
    delta:workflow blyth$ git remote remove origin
    delta:workflow blyth$ git remote -v


log check
---------

::

   git log --name-only 


sizes
-------

* https://confluence.atlassian.com/bitbucket/reduce-repository-size-321848262.html

Filtering only cuts from 62M to 55M. Only after subsequent clone get down to 10M.

::

    delta:workflow blyth$ du -hs /usr/local/env/adm/svn2git/workflow/.git
     62M    /usr/local/env/adm/svn2git/workflow/.git
    delta:workflow blyth$ du -hs .git
     55M    .git
    delta:workflow blyth$ du -hs /tmp/t/workflow/.git
     10M    /tmp/t/workflow/.git

After pruning get down to the same size as the clone::

    delta:workflow blyth$ gitfilter-;gitfilter-prune
    Counting objects: 15726, done.
    ...   
    delta:workflow blyth$ du -hs .git 
     10M    .git


::

    delta:t blyth$ cd workflow/
    delta:workflow blyth$ git_find_big.sh
    All sizes are in kB's. The pack column is the size of the object, compressed, inside the pack file.
    size  pack  SHA                                       location
    7334  2197  22e903e2d95956bd1d1a9ef6734bfbe95f585d60  workflow/build/workflow.build/workflow.pbxindex/symbols0.pbxsymbols
    2240  565   b1d2070b83300ab6361508dcebb1b3b3427eb53f  workflow/build/workflow.build/workflow.pbxindex/strings.pbxstrings/strings
    2100  1214  d28664228135f61610a7af295b6e065a2de08d9a  workflow/build/workflow.build/workflow.pbxindex/cdecls.pbxbtree
    2048  795   97d26dc3583c9fd3ad2eda03fc0839d7a5ffa2ce  workflow/build/workflow.build/workflow.pbxindex/strings.pbxstrings/control
    978   691   d30b3eaaefd0bf2b0ceb3f848e37152832e647b6  workflow/build/workflow.build/workflow.pbxindex/decls.pbxbtree
    736   502   be02ad082dcad1638c415ce32287a3b3382869ac  workflow/build/workflow.build/workflow.pbxindex/refs.pbxbtree
    627   608   5ff8a49bf12515084fc3ecc092f2c66b81994739  notes/sw/lightroom/lightroom-license.png
    204   29    ac0fba8357448e20841ea22cbb691937bac02987  admin/publist/spires-cv-publist.tex
    187   28    e7f51e2f55566bc0d7acf9ca879b390f7325e2c0  admin/pubs/2014/blyth-publist-2014.tex
    152   16    085a188b28a27da4b84ed615b095e877c5fa43f0  workflow.xcodeproj/project.pbxproj



EOU
}

#gitfilter-repo(){ echo workflow_home ; }
gitfilter-repo(){ echo workflow_workflow ; }

gitfilter-target()
{
   case $(gitfilter-repo) in 
     workflow_home)     echo home ;; 
     workflow_workflow) echo workflow ;; 
   esac
}

gitfilter-srcrepo()
{
   case $(gitfilter-repo) in 
     workflow_home)     echo workflow ;; 
     workflow_workflow) echo workflow ;; 
   esac
}


gitfilter-index-filter-note(){ cat << EON

$FUNCNAME
==============================

Switch between which repo to create by changing gitfilter-repo
Which changes the dirs that are dropped.


EON
}


gitfilter-index-filter-(){ 

   local tgt=$(gitfilter-target)
   case $tgt in 
         home) ~/h/census.py --paths --cat drop workflow  2>/dev/null ;; 
     workflow) ~/h/census.py --paths --cat drop home      2>/dev/null ;; 
   esac
}
gitfilter-index-filter(){  cat << EOC
git rm --quiet --cached --ignore-unmatch -r $(echo $($FUNCNAME-)) 
EOC
} 

gitfilter-dir(){  echo $(local-base)/env/adm/gitfilter ; }
gitfilter-srcfold(){  echo $(local-base)/env/adm/svn2git ; }
gitfilter-cd(){   local dir=$(gitfilter-dir) && mkdir -p $dir && cd $dir ;  }

gitfilter-rcd(){ cd $(gitfilter-dir)/$(gitfilter-repo) ; }

gitfilter-info(){ cat << EOI

$FUNCNAME
=================

gitfilter-dir    : $(gitfilter-dir)
gitfilter-repo   : $(gitfilter-repo)
gitfilter-target : $(gitfilter-target)
gitfilter-srcrepo   : $(gitfilter-srcrepo)
gitfilter-srcfold   : $(gitfilter-srcfold)


EOI
}

gitfilter--()
{
   gitfilter-wipe   
      ## remove pre-existing target repo, which has been filtered and pruned already 

   gitfilter-clone  
      ## clone from the svn2git converted repo 
      ## when doing for real will need to make the svn commits and redo the svn2git

   gitfilter-filter
      ## inplace removal of the excluded folders and corresponding history  

   gitfilter-disconnect
      ## disconnect from origin, as have rewritten history 

   gitfilter-prune
      ## shink the repo, removing unreferenced

   gitfilter-find-big
      ## check to find the largest items

   gitfilter-partition-clone
      ## clone into HOME/tgt 
}




gitfilter-index-filter-notes(){ cat << EON
$FUNCNAME
===============================

*gitfilter-index-filter-* function returns a list of top level repo directories
which are excluded from the filtered repo at every repository 
revision.

EON
}


gitfilter-clone()
{
   local msg="=== $FUNCNAME :"
   echo $msg clone svn2hg workflow repo for compartmentalization 
   gitfilter-cd 
   local repo=$(gitfilter-repo)
   local srcrepo=$(gitfilter-srcrepo)
   local srcfold=$(gitfilter-srcfold)

   local cmd="git clone file://$srcfold/$srcrepo $repo "
   [ ! -d $repo ] && echo $cmd && eval $cmd
}

gitfilter-wipe()
{
   local msg="=== $FUNCNAME :"
   gitfilter-cd 
   local repo=$(gitfilter-repo)

   [ ! -d $repo ] && echo $msg PWD $PWD no repo $repo && return 

   local cmd="rm -rf $repo"

   local ans
   read -p "$msg PWD $PWD : enter Y to proceed with : $cmd : " ans
   [ $ans != "Y" ] && echo $msg skip && return 
   echo $msg proceed
   eval $cmd
}




gitfilter-filter()
{
   local msg="=== $FUNCNAME :"

   gitfilter-cd
   local repo=$(gitfilter-repo)
   cd $repo

   echo $msg $(date) PWD $PWD : START 

   git filter-branch --prune-empty --index-filter "$(gitfilter-index-filter)" -- --all

   echo $msg $(date) PWD $PWD : DONE
}


gitfilter-prune()
{
   gitfilter-cd
   local repo=$(gitfilter-repo)
   cd $repo

   rm -rf .git/refs/original/ && git reflog expire --all && git gc --aggressive --prune=now

   git reflog expire --all --expire-unreachable=0
   git repack -A -d
   git prune
}

gitfilter-disconnect()
{
   git remote -v
   git remote remove origin
   git remote -v
}

gitfilter-find-big()
{
   gitfilter-cd
   local repo=$(gitfilter-repo)
   cd $repo

   which git_find_big.sh   
   git_find_big.sh   
}

gitfilter-size()
{
   local msg="=== $FUNCNAME :"
   gitfilter-cd
   local repo=$(gitfilter-repo)
   cd $repo

   echo $msg PWD $PWD
   du -hs .git 
}

gitfilter-partition-clone()
{
   local tgt=$(gitfilter-target)
   gitfilter-partition-clone- $tgt 
}

gitfilter-partition-clone-()
{
   local tgt=${1:-home}
   local msg="=== $FUNCNAME :"
   cd
   [ -d $tgt ] && echo $msg target $tgt exists already : first delete it and then rerun : $FUNCNAME   && return  

   local cmd="git clone file://$(gitfilter-dir)/$(gitfilter-repo) $tgt "

   local ans
   read -p "$msg : enter YES to proceed with : $cmd " ans

   [ "$ans" != "YES" ] && echo $msg skipping && return 
   echo $msg proceeding
   eval $cmd
}

gitfilter-chop-notes(){ cat << EON

$FUNCNAME
======================

Want to in addition to chopping folders, also chop the history 
to just grab the last few commits.  

Brief log::

    delta:workflow_workflow blyth$ git log --oneline --decorate -n 10
    a0e89d8 (HEAD, master) prep for scm-backup-all-as-root on g4pb
    622b9d7 prep for partitioning home from workflow with history, split off and svn2git conversion testing
    feb2b37 prepare to split off the new bitbucket private mercurial workflow repo dirs
    682b2a9 complete 1st pass preparation of home/workflow partitioning
    1e21feb purge/cleanup resulting from workflow/home partitioning progress
    d56300e workflow partitioning cleanups
    c043b04 tidy up during census review, remove some binaries
    692b518 notes and moving towards all relative links
    8aad493 trying to ignore wiki and ticket dirs as not yet ready to commit the translations
    d1e2716 start integrating the Trac translation rst into workflow
    delta:workflow_workflow blyth$ 


After exclude census machinery, the top 2 commits are skipped::

    delta:workflow_workflow blyth$ git log --oneline --decorate -n 5
    7640140 (HEAD, master) complete 1st pass preparation of home/workflow partitioning
    9767f2b purge/cleanup resulting from workflow/home partitioning progress
    554289b workflow partitioning cleanups
    24c4afe tidy up during census review, remove some binaries
    ede0316 notes and moving towards all relative links
    delta:workflow_workflow blyth$ 


Detailed log::

    git log --name-status -n 5

From the detailed log, want all commits before and including the below 
to be dropped from the recent history repo.::

    1e21feb "purge/cleanup resulting from workflow/home partitioning progress"
    9767f2b purge/cleanup resulting from workflow/home partitioning progress 

    ## note that after changing the exclude all the SHA1 have changed


After running the chop, 1st try::

    delta:workflow_workflow blyth$ git log --oneline --decorate
    ac4a032 (HEAD, master) prep for scm-backup-all-as-root on g4pb
    9c102b0 prep for partitioning home from workflow with history, split off and svn2git conversion testing
    79e2084 prepare to split off the new bitbucket private mercurial workflow repo dirs
    875520e complete 1st pass preparation of home/workflow partitioning
    9b08f30 gitfilter-chop-orphan from last 1e21feb


2nd try::

    delta:workflow_workflow blyth$ git log --oneline --decorate
    0ecf4ed (HEAD, master) complete 1st pass preparation of home/workflow partitioning
    3252d74 gitfilter-chop-orphan from last 9767f2b


Created gitfilter-chop by following the instructions from https://git-scm.com/book/en/v2/Git-Tools-Replace

* create an initial commit object as our base point with instructions, 
  then rebase the remaining commits we wish to keep on top of it.

* truncating our recent history down so it’s smaller. We need an overlap so we
  can replace a commit in one with an equivalent commit in the other, so we’re
  going to truncate this to just a few commits.

* We can create our base commit using the commit-tree command, which just takes a
  tree and will give us a brand new, parentless commit object SHA-1 back *orphan*.

* OK, so now that we have a base commit, we can rebase the rest of our history on
  top of that with git rebase --onto. The --onto argument will be the SHA-1 we
  just got back from commit-tree and the rebase point will be the 
  parent of the first commit we want to keep *last*

1st try::

    delta:workflow_workflow blyth$ gitfilter-chop
    == gitfilter-chop : git rebase --onto 9b08f309ba5ca56a83ed7fc833789c12233fc8df 1e21feb
    First, rewinding head to replay your work on top of it...
    Applying: complete 1st pass preparation of home/workflow partitioning 
    Applying: prepare to split off the new bitbucket private mercurial workflow repo dirs
    Applying: prep for partitioning home from workflow with history, split off and svn2git conversion testing
    Applying: prep for scm-backup-all-as-root on g4pb 
    delta:workflow_workflow blyth$ 

2nd try with census excluded::

    delta:workflow_workflow blyth$ gitfilter-;gitfilter-chop
    == gitfilter-chop : git rebase --onto 3252d7425449b358a28c679f5b186aab7c9a49dc 9767f2b
    First, rewinding head to replay your work on top of it...
    Applying: complete 1st pass preparation of home/workflow partitioning 
    delta:workflow_workflow blyth$ 


EON
}


#gitfilter-chop-last(){ echo 1e21feb ; }
gitfilter-chop-last(){ echo 9767f2b ; }

gitfilter-chop-orphan(){  echo $FUNCNAME from last $1 | git commit-tree $1^{tree} ; }
gitfilter-chop()
{
    local msg="== $FUNCNAME :"

    gitfilter-cd
    local repo=$(gitfilter-repo)
    cd $repo

    local last=$(gitfilter-chop-last)
    local orphan=$(gitfilter-chop-orphan $last) 
    local rebase="git rebase --onto $orphan $last  "
    echo $msg $rebase
    eval $rebase
}


   



