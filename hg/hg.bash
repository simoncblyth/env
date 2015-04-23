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

* *adm-vi*


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


log templates
~~~~~~~~~~~~~~

* http://stackoverflow.com/questions/3575189/mercurial-log-with-one-liners

Add log alias to .hgrc::

    [alias]
    shortlog = log --template '{node|short} | {date|isodatesec} | {author|user}: {desc|strip|firstline}\n'

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

