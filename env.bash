
#
#  this is not sustainable 
#    ... move to putting top level precursors here only 
#        with the sub precursors defined  in the name/name.bash file
#
env-home(){     [ -n "$BASH_SOURCE" ] &&  echo $(dirname $BASH_SOURCE) || echo $ENV_HOME ; }
env-vi(){       vi $(env-home)/env.bash ; }
env-lastrev(){  svn- ; svn-lastrev $(env-home) ; }
env-rel(){
  local src=$1
  local rel=${src/$(env-home)\//}
  [ "$rel" == "$src" ] && rel=$(env-home)/
  echo $rel
}

env-sourcelink(){
   local src=${1:-$(env-home)/}
   svn-
   echo env:source:/trunk/$(env-rel $src)@$(svn-lastrev $src)
}

#env-localserver(){ echo http://dayabay.phys.ntu.edu.tw ; }
env-localserver(){ echo http://grid1.phys.ntu.edu.tw:8080 ; }
env-url(){         echo $(env-localserver)/repos/env/trunk ; }
env-email(){       echo blyth@hep1.phys.ntu.edu.tw ; }

cronline-(){    . $(env-home)/base/cronline.bash && cronline-env $* ; }
env-(){         . $(env-home)/env.bash && env-env $* ; }
test-(){        . $(env-home)/test/test.bash       && test-env $* ; }
scponly-(){     . $(env-home)/scponly/scponly.bash && scponly-env $* ; }
invenio-(){     . $(env-home)/invenio/invenio.bash && invenio-env $* ; }
nuwa-(){        . $(env-home)/nuwa/nuwa.bash       && nuwa-env $* ; }
memcheck-(){    . $(env-home)/memcheck/memcheck.bash  && memcheck-env $* ; }
eve-(){         . $(env-home)/eve/eve.bash && eve-env $* ; }
sglv-(){        . $(env-home)/eve/SplitGLView/sglv.bash && sglv-env $* ; }
cmt-(){         . $(env-home)/cmt-/cmt-.bash && cmt-env $* ; }

legacy-(){      . $(env-home)/legacy/legacy.bash && legacy-env $* ; }
sshconf-(){     . $(env-home)/base/sshconf.bash && sshconf-env $* ; }
private-(){     . $(env-home)/base/private.bash && private-env $* ; }

dyw-(){         . $(env-home)/dyw/dyw.bash   && dyw-env   $* ; }
root-(){        . $(env-home)/dyw/root.bash  && root-env  $* ; }

_dyb__(){       . $(env-home)/dyb/dyb__.sh              $* ; }
dyb-(){         . $(env-home)/dyb/dyb.bash  && dyb-env  $* ; }
dybi-(){        . $(env-home)/dyb/dybi.bash && dybi-env $* ; }
dybr-(){        . $(env-home)/dyb/dybr.bash && dybr-env $* ; }
dybt-(){        . $(env-home)/dyb/dybt.bash && dybt-env $* ; }
dybpy-(){       . $(env-home)/dybpy/dybpy.bash && dybpy-env $* ; }
dybsvn-(){      . $(env-home)/dyb/dybsvn.bash && dybsvn-env $* ; }

dtracebuild-(){  . $(env-home)/dtrace/dtracebuild.bash && dtracebuild-env $* ; }

apache2-(){     . $(env-home)/apache/apache2.bash && apache2-env $* ; } 
apache-(){      . $(env-home)/apache/apache.bash && apache-env $* ; } 
apacheconf-(){  . $(env-home)/apache/apacheconf/apacheconf.bash && apacheconf-env $* ; } 

caen-(){        . $(env-home)/caen/caen.bash      && caen-env $* ; } 

base-(){        . $(env-home)/base/base.bash    && base-env $* ; } 
local-(){       . $(env-home)/base/local.bash   && local-env $* ; }
elocal-(){      . $(env-home)/base/local.bash   && local-env $* ; }   ## avoid name clash 
cron-(){        . $(env-home)/base/cron.bash    && cron-env $* ; } 
ebash-(){       . $(env-home)/base/bash.bash    && bash-env $* ; }
sudo-(){        . $(env-home)/base/sudo.bash    && sudo-env $* ; }
mail-(){        . $(env-home)/mail/mail.bash    && mail-env $* ; }

scm-(){         . $(env-home)/scm/scm.bash && scm-env $* ; } 
scm-backup-(){  . $(env-home)/scm/scm-backup.bash && scm-backup-env $* ; } 

unittest-(){    . $(env-home)/unittest/unittest.bash && unittest-env $* ; }
qmtest-(){      . $(env-home)/unittest/qmtest.bash  && qmtest-env  $* ; }
enscript-(){    . $(env-home)/enscript/enscript.bash  && enscript-env  $* ; }



nose-(){         . $(env-home)/nose/nose.bash    && nose-env $* ; }
nosebit-(){      . $(env-home)/nosebit/nosebit.bash    && nosebit-env $* ; }

_nose-(){       . $(env-home)/unittest/nose.bash  && _nose-env  $* ; }
_annobit-(){    . $(env-home)/annobit/annobit.bash  && _annobit-env $* ; }

trac-(){        . $(env-home)/trac/trac.bash && trac-env $* ; } 
tracpreq-(){    . $(env-home)/trac/tracpreq.bash && tracpreq-env $* ; } 
tmacros-(){     . $(env-home)/trac/macros/macros.bash  && macros-env $* ; }

package-(){     . $(env-home)/python/package.bash      && package-env $* ; } 
pkg-(){         . $(env-home)/python/pkg.bash          && pkg-env $* ; }
pypi-(){        . $(env-home)/python/pypi.bash         && pypi-env $* ; }

otrac-(){       . $(env-home)/otrac/otrac.bash     && otrac-env $* ; } 
trac-conf-(){   . $(env-home)/otrac/trac-conf.bash && trac-conf-env $* ; } 
trac-ini-(){    . $(env-home)/otrac/trac-ini.bash  && trac-ini-env  $* ; } 
authzpolicy-(){ . $(env-home)/otrac/authzpolicy.bash && authzpolicy-env $* ; }

svn-(){         . $(env-home)/svn/svn.bash         && svn-env $* ; } 


swig-(){        . $(env-home)/swig/swig.bash       && swig-env $* ; } 
sqlite-(){      . $(env-home)/sqlite/sqlite.bash && sqlite-env $* ; } 

swish-(){       . $(env-home)/swish/swish.bash && swish-env $* ; } 

cvs-(){         . $(env-home)/cvs/cvs.bash && cvs-env $* ; } 



db-(){          . $(env-home)/db/db.bash     && db-env $*     ; }


aberdeen-(){    . $(env-home)/aberdeen/aberdeen.bash && aberdeen-env $* ; }

python-(){      . $(env-home)/python/python.bash  && python-env $*  ; }
ipython-(){     . $(env-home)/python/ipython.bash && ipython-env $* ; }



seed-(){        . $(env-home)/seed/seed.bash && seed-env $* ; }
macros-(){      . $(env-home)/macros/macros.bash && macros-env $* ; }
offline-(){     . $(env-home)/offline/offline.bash && offline-env $* ; }


xml-(){         . $(env-home)/xml/xml.bash ; }





  
# the below may not work in non-interactive running ???  
md-(){  local f=${FUNCNAME/-} && local p=$(env-home)/$f/$f.bash && [ -r $p ] && . $p ; } 
 
 
ee(){ cd $(env-home)/$1 ; }
 
env-usage(){
cat << EOU
#
#     type name        list a function definition 
#     set               list all functions
#     unset -f name     to remove a function
#     typeset -F        lists just the names
#
#  http://www.network-theory.co.uk/docs/bashref/ShellFunctions.html
#  http://www-128.ibm.com/developerworks/library/l-bash-test.html
#
#


     ff(){ local a="hello" ; local ; }   list locals 

     env-dbg
           invoke with bash rather than . when debugging to see 
           line numbers of errors, CAUTION error reporting can be a line off

     env-rsync        top-level-fold <target-node>
           propagate a top-level-folder without svn, caution can
           leave SVN wc state awry ... usually easiest to delete working
           copy and "svn up" when want to come clean and go back to SVN
     
     env-rsync-all    <target-node>
           bootstrapping a node that does not have svn 

     env-again
           delete working copy and checkout again 
     env-u
           update the working copy ... aliased to "eu" 
          
          


EOU
}

env-dbg(){
   bash $(env-home)/env.bash
}

env-env(){
  local msg="=== $FUNCNAME :"
 
  BASE_DBG=0
  SCM_DBG=0
  XML_DBG=0
  SEED_DBG=0
  DYW_DBG=0
  DYB_DBG=0 
  TZERO_DBG=0   ## the interactive/non-interactive switch use for debugging cron/batch issues 

  # 
  #  a better way to debug [-t 0 ] issues is  
  #       env -i bash -c ' whatever '
  #   * -i prevents the env from being passed along 
  #   * single quotes to protech from "this" shell
  #

  local iwd=$(pwd)
  cd $(env-home)
 
  base-  
  svn-
  #scm-    
 
 ## [ "$NODE_TAG" == "H" -o "$NODE_TAG" == "U" ] && export MSG="$msg skipped dyb- on node $NODE_TAG " || dyb- 
 
  cd $iwd
  alias eu=env-u
  
  
  cmt-  
  

}

sss(){
   local msg="=== $FUNCNAME :"
   [ "$NODE_TAG" != "C" ] && echo $msg only on C && return 1
   
   svn-
   svnsync-
   svnsync-synchronize
}


env-u(){ 
  iwd=$(pwd)
  
  if [ "$NODE_TAG" == "$SOURCE_TAG" ]; then
     echo ============= env-u : no svn update is performed as on source node ================
  else
     cd $(env-home) 
     
     echo ============= env-u : status before update ================
     svn status -u
     svn update
     echo ============= env-u : status after update ================
     svn status -u
     cd $iwd
     
  fi
  echo ============== env-u :  sourcing the env =============
  [ -r $(env-home)/env.bash ] && . $(env-home)/env.bash  
}





env-wiki(){ 

[ 1 == 2 ] && cat << EOD	
   #	
   #  usage examples :	
   #
   #	 env-wiki export WikiStart WikiStart
   #           export the wiki page "WikiStart" to file  "WikiStart"
   #   
   #	 env-wiki import WikiStart WikiStart
   #           import the file into the web app
   #
   #
EOD
	 trac-admin $SCM_FOLD/tracs/env wiki $* ; 
}


env-find(){
  q=${1:-dummy}
  cd $(env-home)

  if [ "$(uname)" == "Darwin" ]; then
     mdfind -onlyin . $q
  else
     find . -name '*.*' -exec grep -H $q {} \;
  fi
}





env-x-pkg(){

  X=${1:-$TARGET_TAG}

  if [ "$X" == "ALL" ]; then
    xs="P H G1"
  else
    xs="$X"
  fi

  for x in $xs
  do	
     base-x-pkg $x
     scm-x-pkg $x
     dyw-x-pkg $x
  done

}


env-x-pkg-not-working(){

  X=${1:-$TARGET_TAG}

  iwd=$(pwd)
  cd $(env-home)
  dirs=$(ls -1)
  for d in $dirs
  do
    if [ -d $d ]; then
  		cmd="$d-x-pkg $X"
		echo $cmd
		eval $cmd
 	fi 	
  done

}

env-local-dir(){
   sudo mkdir -p $LOCAL_BASE/env 
   sudo chown $USER $LOCAL_BASE/env
}





env-rsync(){

   local fold=${1:-dybpy}
   local target=${2:-C}
   
   local cmd="rsync  -raztv $(env-home)/$fold/ $target:env/$fold/ --exclude '*.pyc' --exclude '.svn'  "
    
   echo $cmd 
   eval $cmd

}

env-rsync-all(){
  local msg="=== $FUNCNAME :"
  echo $msg to SVNless nodes 
  local tag
  for tag in $* ; do
    env-rsync-all- $tag
  done
}

env-rsync-all-(){
   local msg="=== $FUNCNAME :"
   local target=${1:-H2}
   local cmd="rsync -e ssh  -raztv $(env-home)/ $target:env/ --exclude '*.pyc' --exclude '.svn' --exclude '*.xcodeproj'  "
   echo $cmd 
   [ "$NODE_TAG" == "$target" ] && echo $msg ABORT cannot rsync to self && return 1
   eval $cmd
}


env-path(){
   echo $PATH | tr ":" "\n"
}

env-prepend(){
  local add=$1 
  echo $PATH | grep -v $add - > /dev/null  && export PATH=$add:$PATH 
}

env-append(){
  local add=$1 
  echo $PATH | grep -v $add - > /dev/null && export PATH=$PATH:$add 
}

env-llp-prepend(){
  local add=$1 
  
  if [ "$(uname)" == "Darwin" ]; then
    echo $DYLD_LIBRARY_PATH | grep -v $add - > /dev/null && export DYLD_LIBRARY_PATH=$add:$DYLD_LIBRARY_PATH
  else  
    echo $LD_LIBRARY_PATH | grep -v $add - > /dev/null && export LD_LIBRARY_PATH=$add:$LD_LIBRARY_PATH 
  fi
}

env-pp-prepend(){
  local add=$1 
  echo $PYTHONPATH | grep -v $add - > /dev/null && export PYTHONPATH=$add:$PYTHONPATH
}

env-pp(){
  echo $PYTHONPATH | tr ":" "\n" 
}


env-llp(){
   if [ "$(uname)" == "Darwin" ]; then
      echo $DYLD_LIBRARY_PATH | tr ":" "\n"
   else
      echo $LD_LIBRARY_PATH | tr ":" "\n"
   fi    
}


env-ldconfig(){

   local msg="=== $FUNCNAME :"
   local dir=$1
   
   local conf=/etc/ld.so.conf
   
   [ ! -d $dir  ]                && echo $msg ABORT no dir $dir && return 1
   [ ! -f $conf ]                && echo $msg ABORT no $conf && return 1
   [ "$(which ldconfig)" == "" ] && echo $msg ABORT no ldconfig in your path && return 1
   
   grep -q $dir $conf && echo $msg the dir $dir is already there $conf && return 0
   
   echo $msg appending $dir to $conf
   sudo bash -c "echo $dir >> $conf  " 

   cat $conf
   sudo ldconfig
   
}

env-ldconf(){

   local conf=/etc/ld.so.conf
   cat $conf 
   


}




env-again(){

  local msg="=== $FUNCNAME :"

  [ -z $(env-home) ] && echo $msg ABORT no $(env-home) && return 1 
  
  local dir=$(dirname $(env-home))
  local name=$(basename $(env-home))
  local url=$(env-url)
  
  read -p "$msg are you sure you want to wipe $name from $dir and then checkout again from $url  ? answer YES to proceed "  answer
  [ "$answer" != "YES" ] && echo $msg skipping && return 1 
  
  cd $dir && rm -rf $name && svn co $url $name

}


env-egglink(){

   cat << DELIB
   
   setuptools needs layout .,..
   
       EnvDistro
          setup.py
          env/
             __init__.py 
             trac/
                __init__.py
   
   
   
DELIB

  local msg="=== $FUNCNAME :" 
  local dir=$(dirname $(env-home))
  cd $dir
  local setup="setup.py"
  [ -f "$setup" ] && echo $msg a $setup exists already in $dir, delete $setup and rerun && return 1

  env-egglink-setup > $setup

  echo $msg proceed to egglink using setuptools develop mode
  which python
  python setup.py develop
  

}


env-egglink-setup(){

   local tlpkgs="$*"
   cat << EOS
"""
   This was sourced from $BASH_SOURCE:$FUNCNAME at $(date)

   The find_packages is going to be very user specific... 
   as it depends on what exists above $(env-home)

"""
from setuptools import setup, find_packages

pkgs = find_packages(exclude=['hz*','e','e.*','w','w.*'])
print "egglinking the packages.. %s " % pkgs

setup(
      name='Env',
      version='0.1',
      packages = pkgs ,
      )
   
   
EOS

}




env-env

