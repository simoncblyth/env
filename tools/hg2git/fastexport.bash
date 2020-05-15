# === func-gen- : tools/hg2git/fastexport fgp tools/hg2git/fastexport.bash fgn fastexport fgh tools/hg2git
fastexport-src(){      echo tools/hg2git/fastexport.bash ; }
fastexport-source(){   echo ${BASH_SOURCE:-$(env-home)/$(fastexport-src)} ; }
fastexport-vi(){       vi $(fastexport-source) ; }
fastexport-env(){      elocal- ; bitbucket- ; }
fastexport-usage(){ cat << EOU

fastexport
=============


See Also
----------

* hgexporttool-



May 2020 issues
-----------------

::


    NOT WITH THE _HG NAMES

    === fastexport-hg2git : opticks -> opticks_git : log opticks_git.log PWD /usr/local/env/tools/hg2git/fast-export.operations
    === fastexport-hg2git : opticks -> opticks_git : RC 0
    === fastexport-hg2git : env -> env_git : log env_git.log PWD /usr/local/env/tools/hg2git/fast-export.operations
    === fastexport-hg2git : env -> env_git : RC 0

    THE BELOW ERRS WHERE FROM NETWORK GOING DOWN

    === fastexport-hg2git : simoncblyth.bitbucket.io -> simoncblyth.bitbucket.io_git : log simoncblyth.bitbucket.io_git.log PWD /usr/local/env/tools/hg2git/fast-export.operations
    === fastexport-hg2git : simoncblyth.bitbucket.io -> simoncblyth.bitbucket.io_git : RC 1
    === fastexport-hg2git : simoncblyth.bitbucket.io -> simoncblyth.bitbucket.io_git : ERR
    === fastexport-hg2git : implicitmesher -> implicitmesher_git : log implicitmesher_git.log PWD /usr/local/env/tools/hg2git/fast-export.operations
    === fastexport-hg2git : implicitmesher -> implicitmesher_git : RC 1
    === fastexport-hg2git : implicitmesher -> implicitmesher_git : ERR
    === fastexport-hg2git : opticksdata -> opticksdata_git : log opticksdata_git.log PWD /usr/local/env/tools/hg2git/fast-export.operations
    === fastexport-hg2git : opticksdata -> opticksdata_git : RC 1
    === fastexport-hg2git : opticksdata -> opticksdata_git : ERR



May 2020
------------

* https://repo.or.cz/fast-export.git

* https://github.com/chrisjbillington/hg-export-tool
* https://github.com/chrisjbillington/hg-export-tool/blob/master/exporter.py

hg heads
~~~~~~~~~~~

::

   --closed
         show normal and closed branch heads
   --topo
         show topological heads only (changesets with no children)  


::

    epsilon:env blyth$ hg heads --closed --topo --template json
    [
     {
      "rev": 6445,
      "node": "7c74c85d044484f0a5fa695a63a6d123bcd53a14",
      "branch": "default",
      "phase": "public",
      "user": "Simon Blyth <simoncblyth@gmail.com>",
      "date": [1586878901, -3600],
      "desc": "misc",
      "bookmarks": [],
      "tags": ["tip"],
      "parents": ["a2c80cef597f59674777397fab4ee49a0170641e"]
     }
    ]
    epsilon:env blyth$ hg heads --closed --template json
    [
     {
      "rev": 6445,
      "node": "7c74c85d044484f0a5fa695a63a6d123bcd53a14",
      "branch": "default",
      "phase": "public",
      "user": "Simon Blyth <simoncblyth@gmail.com>",
      "date": [1586878901, -3600],
      "desc": "misc",
      "bookmarks": [],
      "tags": ["tip"],
      "parents": ["a2c80cef597f59674777397fab4ee49a0170641e"]
     }
    ]

    ## huh UTC-1hr ?  locale noticed BST british-summer-time ?



What is a topological head ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.mercurial-scm.org/wiki/UnderstandingMercurial




* https://dzone.com/articles/convert-a-mercurial-repository-to-git-using-hg-fas


16 bitbucket repos :  13 are mercurial in need of conversion
-------------------------------------------------------------

* https://bitbucket.org/simoncblyth/workspace/repositories


opticks
    hg 
simoncblyth.bitbucket.io
    hg 
env
    hg 
implicitmesher
    hg
opticksdata
    hg 
g4dae
    hg
g4dae-opticks
    hg 
heprez
    hg

intro_to_numpy
    hg
intro_to_cuda
    hg 

jnu
    hg 
tracdev
    hg, small : dont care about loss : good to practice on 
chroma
    hg, fork of chroma/chroma



scenekittest
    git
meshlab
    git, started from a tarball 
opticksaux
    git








* https://stackoverflow.com/questions/10710250/converting-mercurial-folder-to-a-git-repository

* http://repo.or.cz/w/fast-export.git

::

    mkdir repo-git # or whatever
    cd repo-git
    git init
    hg-fast-export.sh -r <local-repo>
    git checkout HEAD


::

    Initialized empty Git repository in /Users/blyth/env_git/.git/
    Error: The option core.ignoreCase is set to true in the git
    repository. This will produce empty changesets for renames that just
    change the case of the file name.
    Use --force to skip this check or change the option with
    git config core.ignoreCase false
    delta:env_git blyth$ 



EOU
}
fastexport-dir(){ echo $(local-base)/env/tools/hg2git/fast-export ; }
fastexport-cd(){  cd $(fastexport-dir); }

fastexport-odir(){ echo $(local-base)/env/tools/hg2git/fast-export.operations ; }
fastexport-ocd(){  local odir=$(fastexport-odir) ; mkdir -p $odir ; cd $odir ;  }

fastexport-get-notes(){ cat << EON

Note: checking out 'tags/v180317'.

You are in 'detached HEAD' state. You can look around, make experimental
changes and commit them, and you can discard any commits you make in this
state without impacting any branches by performing another checkout.

If you want to create a new branch to retain commits you create, you may
do so (now or later) by using -b with the checkout command again. Example:

  git checkout -b <new-branch-name>

HEAD is now at 19aa906... Update usage section example commands



* https://github.com/frej/fast-export/releases
* v180317 :  frej released this on Mar 17, 2018 

Releases after v180317 pin to particular Mercurial versions
which is not very useful.


EON
}


fastexport-get(){

   local iwd=$PWD
   local dir=$(dirname $(fastexport-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d fast-export ] && git clone git://repo.or.cz/fast-export.git  
 
   cd fast-export
   git checkout tags/v180317

   cd $iwd

}




fastexport-hg2git-notes(){ cat << EON


With macports mercurial 5.3_1 and fast_export master
------------------------------------------------------

Could not find a python interpreter with the mercurial module >= 4.6 available.  
Please use the 'PYTHON' environment variable to specify the interpreter to use.::

    In [11]: import mercurial.scmutil

    In [12]: mercurial.__file__
    Out[12]: '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/mercurial/__init__.pyc'

    epsilon:tracdev_git blyth$ python -c "from mercurial.scmutil import revsymbol"
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ImportError: cannot import name revsymbol

    port info mercurial
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
    mercurial @5.3_1 (devel, python)
    Sub-ports:            mercurial-devel
    Variants:             bash_completion, universal, zsh_completion


* https://github.com/frej/fast-export/pull/131



With macports mercurial 5.3_1 and fast_export tags/v180317
-----------------------------------------------------------------

::

    epsilon:tracdev_git blyth$ python -c "from mercurial import hg"
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/mercurial/hg.py", line 21, in <module>
        from . import (
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/mercurial/bundlerepo.py", line 23, in <module>
        from . import (
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/mercurial/cmdutil.py", line 24, in <module>
        from . import (
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/mercurial/crecord.py", line 30, in <module>
        locale.setlocale(locale.LC_ALL, u'')
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/locale.py", line 581, in setlocale
        return _setlocale(category, locale)
    locale.Error: unsupported locale setting

    epsilon:tracdev_git blyth$ export LC_ALL="en_US.UTF-8"
    epsilon:tracdev_git blyth$ export LC_CTYPE="en_US.UTF-8"
    epsilon:tracdev_git blyth$ 
    epsilon:tracdev_git blyth$ python -c "from mercurial import hg"
    epsilon:tracdev_git blyth$ 


Rename repos in bitbucket web interface appending _hg 
-------------------------------------------------------


* https://stackoverflow.com/questions/57716968/convert-bitbucket-mercurial-repository-to-git-maintain-branches-and-history-on

Procedure:

1. in bitbucket settings simply change the name to eg tracdev_hg 


# https://bitbucket.org/dashboard/repositories


EON
}

# these are hg repos that are not yet renamed _hg 
fastexport-crucial-repolist-(){ cat << EOR
opticks
simoncblyth.bitbucket.io
opticksdata
EOR
}

fastexport-repolist-(){ cat << EOR
env_hg
EOR
}

fastexport-repolist-migrated(){ cat << EOR
tracdev_hg
mountains_hg
intro_to_cuda_hg
intro_to_numpy_hg
heprez_hg
sphinxtest_hg
jnu_hg
g4dae_hg
g4dae-opticks_hg
opticks-cmake-overhaul_hg
chroma_hg
implicitmesher_hg
EOR
}

fastexport-originally-git-repolist-(){ cat << EOR
opticksaux
workflow
play
scenekittest
meshlab
EOR
}

fastexport-repolist()
{
    fastexport-repolist-
   # fastexport-crucial-repolist-
}

fastexport-authors-all()
{
    local repolist=$(fastexport-repolist)
    local repo 
    for repo in $repolist ; do 
        fastexport-authors $repo  
    done 
}
fastexport-authors()
{
   local name_hg=$1
   fastexport-ocd 
   cd $name_hg 
   printf "\n ############ %s ################# \n\n" $name_hg 
   hg log --template "{author}\n" | sort | uniq
}



# fastexport-;fastexport-hg2git-all
fastexport-hg2git-all()
{
    local repolist=$(fastexport-repolist)
    local repo 
    for repo in $repolist ; do 
        fastexport-hg2git $repo  
    done 
}



fastexport-hg2git-notes(){ cat << EON
fastexport-hg2git
===================

::

   fastexport-hg2git reponame_hg 
       single argument must be the name of Mercurial repo 
       and must end with "_hg" 


EON
}


fastexport-hg2git()
{
   local name_hg=${1:-tracdev_hg}
   local name=$(fastexport-namegit $name_hg)
   local msg="=== $FUNCNAME : $name_hg -> $name :"
   fastexport-ocd 

   [ -d "$name" -a -d "$name/.git" ] && echo $msg $name already converted to git && return 0

   local log=$name.log
   echo $msg log $log PWD $PWD  

   fastexport-hg2git- $name_hg 1>$log 2>&1
   rc=$?
   echo $msg RC $rc
   [ $rc -ne 0 ] && echo $msg ERR && return 1
   return 0  
}


fastexport-namegit()
{
   local name_hg=$1
   local stem=${name_hg/_hg}   
   local name
   if [ "${name_hg:(-3)}" == "_hg" ]; then 
       name=$stem
   else
       name=${stem}_git
   fi 
   echo $name
}

fastexport-hg2git-()
{
   local check=$(basename $PWD)
   [ "$check" != "$(basename $(fastexport-odir))" ] && echo $msg ERROR must invoke $FUNCNAME from operations dir not $PWD && return 5

   local iwd=$PWD

   local name_hg=${1:-tracdev_hg}
   local name=$(fastexport-namegit $name_hg)
   local msg="=== $FUNCNAME $name_hg -> $name :"
   echo $msg DATE START $(date) 

   local urlhg=$(bitbucket-urlhg $name_hg)

   # rerun 
   if [ ! -d "${name_hg}" ]; then
       echo $msg cloning from $url pwd $PWD
       hg clone $urlhg 
   else
       pushd ${name_hg} > /dev/null
       hg update 
       [ $? -ne 0 ] && echo $msg ERR from update && return 1
       popd > /dev/null
       echo $msg repo from $url already cloned pwd $PWD 
   fi  

   [ ! -d "${name_hg}" ]     && echo $msg ERROR no such dir ${name_hg} && return 1 
   [ ! -d "${name_hg}/.hg" ] && echo $msg ERROR no .hg dir  && return 2 


   echo $msg unique authors [
   pushd $name_hg > /dev/null
   hg log --template "{author}\n" | sort | uniq
   popd > /dev/null
   echo $msg unique authors ]

   local rc=0
   local script=$(fastexport-dir)/hg-fast-export.sh
   [ ! -f "$script" ] && echo $msg script $script does not exist && return 3

   # rerunning deletes the git repo and converts again
   rm -rf $name
   mkdir $name
   local urlgit=$(bitbucket-urlgit $name)
   pushd $name > /dev/null

      git init
      git config core.ignoreCase false

      $script -r ../${name_hg}
      rc=$?
      echo $msg rc from conversion $rc 
      [ $rc -ne 0 ] && echo $msg non-zero RC from conversion && cd $iwd && return $rc 

      git checkout HEAD

      #bitbucket-git-remote
      git remote add origin $urlgit
      git remote -v
      echo $msg push to remote with : git push -u origin master

   popd > /dev/null
   echo $msg DATE STOP $(date) 
   return $rc
}


fastexport-push-to-remote()
{
    echo see bitbucket-add-remote
}


fastexport-env-check(){ cat << EON

delta:~ blyth$ cd env
delta:env blyth$ pyc
=== clui-pyc : in hg/svn/git repo : remove pyc beneath root /Users/blyth/env
delta:env blyth$ cd ..
delta:~ blyth$ diff -r --brief env env_git

delta:env_git blyth$ git shortlog -e -s -n
   
   ..list of email addesses and commit counts
   ..observe lots of near dupe email addresses
   ..mapping mapping can cleanup

elta:env_git blyth$ 

delta:~ blyth$ du -hs env env_git
125M    env
127M    env_git



EON
}



