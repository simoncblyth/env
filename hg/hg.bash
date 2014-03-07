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



