# === func-gen- : adm/gitfilter fgp adm/gitfilter.bash fgn gitfilter fgh adm
gitfilter-src(){      echo adm/gitfilter.bash ; }
gitfilter-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gitfilter-src)} ; }
gitfilter-vi(){       vi $(gitfilter-source) ; }
gitfilter-env(){      elocal- ; }
gitfilter-usage(){ cat << EOU

gitfilter : using filter-branch subcommand
===========================================

Extract some folders from a git repo into a new one
-----------------------------------------------------

* https://confluence.atlassian.com/bitbucket/split-a-repository-in-two-313464964.html
* https://help.github.com/articles/splitting-a-subfolder-out-into-a-new-repository/
* https://stackoverflow.com/questions/359424/detach-move-subdirectory-into-separate-git-repository/17864475#17864475
* https://stackoverflow.com/questions/359424/detach-move-subdirectory-into-separate-git-repository
* https://stackoverflow.com/questions/2982055/detach-many-subdirectories-into-a-new-separate-git-repository

::

    git filter-branch --index-filter "git rm -r -f --cached --ignore-unmatch $(ls -xd apps/!(AAA) libs/!(XXX))" --prune-empty -- --all


* https://www.atlassian.com/blog/git/tear-apart-repository-git-way


TODO
------

* peruse history to check what else can be removed 
* check repo sizes, try some shrinkage : repacking 


gitfilter-filter
-----------------

::

    delta:gitfilter blyth$ gitfilter-;gitfilter-filter
    === gitfilter-test : Fri Mar 9 22:58:13 CST 2018 PWD /usr/local/env/adm/gitfilter/workflow : START
    Rewrite b20eadf16b36edf9efc175881c4665390b9360d5 (1213/1213)
    Ref 'refs/heads/master' was rewritten
    Ref 'refs/remotes/origin/master' was rewritten
    WARNING: Ref 'refs/remotes/origin/master' is unchanged
    === gitfilter-test : Fri Mar 9 22:59:01 CST 2018 PWD /usr/local/env/adm/gitfilter/workflow : DONE
    delta:workflow blyth$ 



Rewriting means are changing entrire history so the connection with origin
is broken, or at least weakened::

    delta:workflow blyth$ git remote -v
    origin  file:///usr/local/env/adm/svn2git/workflow (fetch)
    origin  file:///usr/local/env/adm/svn2git/workflow (push)
    delta:workflow blyth$ git remote remove origin
    delta:workflow blyth$ git remote -v


clone it into new home
------------------------

::

    delta:gitfilter blyth$ git clone file://$PWD/workflow home 
    Cloning into 'home'...
    remote: Counting objects: 10879, done.
    remote: Compressing objects: 100% (3524/3524), done.
    remote: Total 10879 (delta 6688), reused 10879 (delta 6688)
    Receiving objects: 100% (10879/10879), 3.77 MiB | 0 bytes/s, done.
    Resolving deltas: 100% (6688/6688), done.
    Checking connectivity... done.
    delta:gitfilter blyth$ 


wdocs and Sphinx build need to jump ship from workflow to home
----------------------------------------------------------------- 



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


Hmm one of the first commits was inadvertent::

    delta:workflow blyth$ git log --name-only -- workflow
    commit 426ba7eae67c84843913d480a694e472c3b6fda2
    Author: Simon C Blyth <simon.c.blyth@gmail.com>
    Date:   Sat Jun 16 09:14:27 2007 +0000

        remove bad xcodeproj

    workflow/build/workflow.build/workflow.pbxindex/categories.pbxbtree
    workflow/build/workflow.build/workflow.pbxindex/cdecls.pbxbtree
    ..

    commit d9e26d24cbeaa2268da37bb7df9523d1e1c6bb90
    Author: Simon C Blyth <simon.c.blyth@gmail.com>
    Date:   Fri Jun 15 15:51:11 2007 +0000

        initial scm-import

    workflow/build/workflow.build/workflow.pbxindex/categories.pbxbtree
    workflow/build/workflow.build/workflow.pbxindex/cdecls.pbxbtree
    ..



EOU
}

gitfilter-repo(){ echo workflow ; }
gitfilter-dir(){  echo $(local-base)/env/adm/gitfilter ; }
gitfilter-srcr(){  echo $(local-base)/env/adm/svn2git ; }
gitfilter-cd(){   local dir=$(gitfilter-dir) && mkdir -p $dir && cd $dir ;  }

gitfilter-rcd(){ cd $(gitfilter-dir)/$(gitfilter-repo) ; }

gitfilter-info(){ cat << EOI

$FUNCNAME
=================

gitfilter-dir  : $(gitfilter-dir)
gitfilter-repo : $(gitfilter-repo)
gitfilter-srcr : $(gitfilter-srcr)


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

   gitfilter-home
      ## clone into HOME/home 
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
   local srcr=$(gitfilter-srcr)

   local cmd="git clone file://$srcr/$repo $repo "
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



gitfilter-index-filter-(){ ~/w/census.py --dirs --cat drop workflow  2>/dev/null ; }
gitfilter-index-filter(){  cat << EOC
git rm --quiet --cached --ignore-unmatch -r $(echo $($FUNCNAME-)) 
EOC
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


gitfilter-home()
{
   local msg="=== $FUNCNAME :"
   cd
   [ -d home ] && echo $msg home exists already : first delete it and then rerun : $FUNCNAME   && return  
   git clone file://$(gitfilter-dir)/workflow home 
}



