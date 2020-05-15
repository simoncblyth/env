# === func-gen- : tools/hg2git/fastexport fgp tools/hg2git/fastexport.bash fgn fastexport fgh tools/hg2git
fastexport-src(){      echo tools/hg2git/fastexport.bash ; }
fastexport-source(){   echo ${BASH_SOURCE:-$(env-home)/$(fastexport-src)} ; }
fastexport-vi(){       vi $(fastexport-source) ; }
fastexport-env(){      elocal- ; }
fastexport-usage(){ cat << EOU

fastexport
=============


See Also
----------

* hgexporttool-



May 2020 issues
-----------------

epsilon:fast-export.operations blyth$ fastexport-;fastexport-hg2git-all
=== fastexport-hg2git : tracdev_hg -> tracdev : log tracdev.log PWD /usr/local/env/tools/hg2git/fast-export.operations
=== fastexport-hg2git : tracdev_hg -> tracdev : RC 0
=== fastexport-hg2git : chroma_hg -> chroma : log chroma.log PWD /usr/local/env/tools/hg2git/fast-export.operations
=== fastexport-hg2git : chroma_hg -> chroma : RC 0
=== fastexport-hg2git : g4dae_hg -> g4dae : log g4dae.log PWD /usr/local/env/tools/hg2git/fast-export.operations
=== fastexport-hg2git : g4dae_hg -> g4dae : RC 0
=== fastexport-hg2git : g4dae-opticks_hg -> g4dae-opticks : log g4dae-opticks.log PWD /usr/local/env/tools/hg2git/fast-export.operations
=== fastexport-hg2git : g4dae-opticks_hg -> g4dae-opticks : RC 0
=== fastexport-hg2git : heprez_hg -> heprez : log heprez.log PWD /usr/local/env/tools/hg2git/fast-export.operations
=== fastexport-hg2git : heprez_hg -> heprez : RC 0
=== fastexport-hg2git : intro_to_cuda_hg -> intro_to_cuda : log intro_to_cuda.log PWD /usr/local/env/tools/hg2git/fast-export.operations
=== fastexport-hg2git : intro_to_cuda_hg -> intro_to_cuda : RC 0
=== fastexport-hg2git : intro_to_numpy_hg -> intro_to_numpy : log intro_to_numpy.log PWD /usr/local/env/tools/hg2git/fast-export.operations
=== fastexport-hg2git : intro_to_numpy_hg -> intro_to_numpy : RC 0
=== fastexport-hg2git : jnu_hg -> jnu : log jnu.log PWD /usr/local/env/tools/hg2git/fast-export.operations
=== fastexport-hg2git : jnu_hg -> jnu : RC 0
=== fastexport-hg2git : mountains_hg -> mountains : log mountains.log PWD /usr/local/env/tools/hg2git/fast-export.operations
=== fastexport-hg2git : mountains_hg -> mountains : RC 0
=== fastexport-hg2git : opticks-cmake-overhaul_hg -> opticks-cmake-overhaul : log opticks-cmake-overhaul.log PWD /usr/local/env/tools/hg2git/fast-export.operations
=== fastexport-hg2git : opticks-cmake-overhaul_hg -> opticks-cmake-overhaul : RC 0
=== fastexport-hg2git : sphinxtest_hg -> sphinxtest : log sphinxtest.log PWD /usr/local/env/tools/hg2git/fast-export.operations
=== fastexport-hg2git : sphinxtest_hg -> sphinxtest : RC 0
=== fastexport-hg2git : opticks -> opticks_git : log opticks_git.log PWD /usr/local/env/tools/hg2git/fast-export.operations
=== fastexport-hg2git : opticks -> opticks_git : RC 0
=== fastexport-hg2git : env -> env_git : log env_git.log PWD /usr/local/env/tools/hg2git/fast-export.operations
=== fastexport-hg2git : env -> env_git : RC 0
=== fastexport-hg2git : simoncblyth.bitbucket.io -> simoncblyth.bitbucket.io_git : log simoncblyth.bitbucket.io_git.log PWD /usr/local/env/tools/hg2git/fast-export.operations
=== fastexport-hg2git : simoncblyth.bitbucket.io -> simoncblyth.bitbucket.io_git : RC 1
=== fastexport-hg2git : simoncblyth.bitbucket.io -> simoncblyth.bitbucket.io_git : ERR
=== fastexport-hg2git : implicitmesher -> implicitmesher_git : log implicitmesher_git.log PWD /usr/local/env/tools/hg2git/fast-export.operations
=== fastexport-hg2git : implicitmesher -> implicitmesher_git : RC 1
=== fastexport-hg2git : implicitmesher -> implicitmesher_git : ERR
=== fastexport-hg2git : opticksdata -> opticksdata_git : log opticksdata_git.log PWD /usr/local/env/tools/hg2git/fast-export.operations
=== fastexport-hg2git : opticksdata -> opticksdata_git : RC 1
=== fastexport-hg2git : opticksdata -> opticksdata_git : ERR
epsilon:fast-export.operations blyth$ 




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

fastexport-crucial-repolist-(){ cat << EOR
opticks
env
simoncblyth.bitbucket.io
implicitmesher
opticksdata
EOR
}

fastexport-repolist-(){ cat << EOR
tracdev_hg
chroma_hg
g4dae_hg
g4dae-opticks_hg
heprez_hg
intro_to_cuda_hg
intro_to_numpy_hg
jnu_hg
mountains_hg
opticks-cmake-overhaul_hg
sphinxtest_hg
EOR
}


fastexport-git-repolist-(){ cat << EOR
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
    fastexport-crucial-repolist-
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

   local url=ssh://hg@bitbucket.org/simoncblyth/${name_hg}

   # rerun 
   if [ ! -d "${name_hg}" ]; then
       echo $msg cloning from $url pwd $PWD
       hg clone $url 
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
   pushd $name > /dev/null

      git init
      git config core.ignoreCase false

      $script -r ../${name_hg}
      rc=$?
      echo $msg rc from conversion $rc 
      [ $rc -ne 0 ] && echo $msg non-zero RC from conversion && cd $iwd && return $rc 

      git checkout HEAD

   popd > /dev/null
   echo $msg DATE STOP $(date) 
   return $rc
}


fastexport-push-to-remote()
{
    # 1st create the remote git repo, then 
    git remote add origin git@my-git-server:my-repository.git
    git push -u origin master
}


fastexport-env-repo()
{
   cd 
   rm -rf env_git
   mkdir env_git
   cd env_git

   git init
   git config core.ignoreCase false

   $(fastexport-dir)/hg-fast-export.sh -r ../env


}
fastexport-env-repo-notes(){ cat << EON

master: Exporting simple delta revision 6303/6307 with 1/3/0 added/changed/removed files
master: Exporting simple delta revision 6304/6307 with 1/4/1 added/changed/removed files
master: Exporting simple delta revision 6305/6307 with 1/8/0 added/changed/removed files
master: Exporting simple delta revision 6306/6307 with 1/4/0 added/changed/removed files
master: Exporting simple delta revision 6307/6307 with 0/2/0 added/changed/removed files
Issued 6307 commands
git-fast-import statistics:
---------------------------------------------------------------------
Alloc'd objects:      65000
Total objects:        64290 (      3572 duplicates                  )
      blobs  :        29870 (      3212 duplicates      14368 deltas of      29706 attempts)
      trees  :        28113 (       360 duplicates      23512 deltas of      26026 attempts)
      commits:         6307 (         0 duplicates          0 deltas of          0 attempts)
      tags   :            0 (         0 duplicates          0 deltas of          0 attempts)
Total branches:           1 (         1 loads     )
      marks:        1048576 (      6307 unique    )
      atoms:           5958
Memory total:          5219 KiB
       pools:          2173 KiB
     objects:          3046 KiB
---------------------------------------------------------------------
pack_report: getpagesize()            =       4096
pack_report: core.packedGitWindowSize = 1073741824
pack_report: core.packedGitLimit      = 8589934592
pack_report: pack_used_ctr            =      33828
pack_report: pack_mmap_calls          =      18444
pack_report: pack_open_windows        =          1 /          1
pack_report: pack_mapped              =   85599855 /   85599855
---------------------------------------------------------------------


delta:~ blyth$ cd env
delta:env blyth$ pyc
=== clui-pyc : in hg/svn/git repo : remove pyc beneath root /Users/blyth/env
delta:env blyth$ cd ..
delta:~ blyth$ diff -r --brief env env_git
Only in env_git: .git
Only in env: .hg
Only in env: _build
diff: env/bin/G4DAEChromaTest.sh: No such file or directory
diff: env_git/bin/G4DAEChromaTest.sh: No such file or directory
diff: env/bin/cfg4.sh: No such file or directory
diff: env_git/bin/cfg4.sh: No such file or directory
diff: env/bin/doctree.py: No such file or directory
diff: env_git/bin/doctree.py: No such file or directory
diff: env/bin/ggv.py: No such file or directory
diff: env_git/bin/ggv.py: No such file or directory
diff: env/bin/ggv.sh: No such file or directory
diff: env_git/bin/ggv.sh: No such file or directory
Only in env/bin: realpath
Only in env/boost/basio: netapp
Only in env/boost/basio: numpyserver
Only in env/boost/basio: udp_server
Only in env/cuda: optix
Only in env/doc: docutils
Only in env/doc: sphinxtest
Files env/env.bash and env_git/env.bash differ
Only in env/graphics/assimp/AssimpTest: build
Only in env/graphics/isosurface: AdaptiveDualContouring
Only in env/graphics/isosurface: ImplicitMesher
Only in env/graphics: oglrap
Only in env/messaging: js
Only in env/network/asiozmq: examples
Only in env/network/gputest: .gputest.bash.swp
Only in env/npy: quartic
Only in env/numerics: npy
Only in env_git/nuwa/MockNuWa: MockNuWa.cc
Only in env/nuwa/MockNuWa: mocknuwa.cc
Only in env: ok
Only in env/optix/lxe: .cfg4.bash.swp
Only in env/presentation: intro_to_cuda
Only in env/tools: hg2git
delta:~ blyth$ 


delta:~ blyth$ diff env/env.bash env_git/env.bash
1938d1937
< fastexport-(){      . $(env-home)/tools/hg2git/fastexport.bash && fastexport-env $* ; }
delta:~ blyth$ 


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



