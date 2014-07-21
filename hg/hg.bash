# === func-gen- : hg/hg fgp hg/hg.bash fgn hg fgh hg
hg-src(){      echo hg/hg.bash ; }
hg-source(){   echo ${BASH_SOURCE:-$(env-home)/$(hg-src)} ; }
hg-vi(){       vi $(hg-source) ; }
hg-env(){      elocal- ; }
hg-usage(){
  cat << EOU
Mercurial
===========

Tips
----

*hg -v help log*
     gives much more detailed help with the *-v*

*hg log -v -l5*
     *-v* option lists changed files

*hg log -vGl 5*
     combine options, G shows DAG, l to limit revisions 


hg convert
------------

#. needs *sudo port install subversion-python27bindings*
#. authormap
#. timezone

* http://hgbook.red-bean.com/read/migrating-to-mercurial.html






env repo took about 8 minutes over network to D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


manual history comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hg trailing 2 (Hg 4642 svn 4644)::

    delta:env blyth$ hg par
    changeset:   4642:d7570fd518b3
    tag:         tip
    user:        blyth
    date:        Mon Jul 21 20:46:21 2014 +0800
    summary:     mercurial notes


Hg starts 1 behind, due to restricting to trunk alone::

    delta:env blyth$ hg log -r 0
    changeset:   0:9f2fcef8ee0d
    user:        blyth
    date:        Sat May 05 10:36:52 2007 +0800
    summary:     initial import from dummy

    delta:env blyth$ svn log . -r1 -v
    ------------------------------------------------------------------------
    r1 | blyth | 2007-05-05 10:36:52 +0800 (Sat, 05 May 2007) | 1 line
    Changed paths:
       A /branches
       A /tags
       A /trunk

    initial import from dummy 


Something funny about changeset 10

* http://dayabay.phys.ntu.edu.tw/tracs/env/changeset/10 it gives permission denied

::

    delta:env blyth$ svn log . -r10 -v
    ------------------------------------------------------------------------








systematic history checking 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://svn.apache.org/repos/asf/subversion/trunk/tools/examples/
* hgapi



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
        env, heprez, tracdev, workflow
     




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

hg-backup(){
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



