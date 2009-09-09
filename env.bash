


env-home(){     [ -n "$BASH_SOURCE" ] &&  echo $(dirname $BASH_SOURCE) || echo $ENV_HOME ; }
env-source(){   echo $(env-home)/env.bash ; }
env-vi(){       vi $(env-source) ; }
env-ini(){      . $(env-source) ; }
env-lastrev(){  svn- ; svn-lastrev $(env-home) ; }
env-override-path(){ echo $(env-home)/local/$(uname -n).bash ; }
env-rel(){
  local src=$1
  local rel=${src/$(env-home)\//}
  [ "$rel" == "$src" ] && rel=$(env-home)/
  echo $rel
}

env-owners-(){ cat << EOO
    aberdeen blyth
    base     blyth
    scm      blyth
    trac     blyth 
    apache   blyth
    mysql    blyth
    python   blyth
    root     blyth 
    offline  blyth 
EOO
## used by autocomp-owners-
}

env-sourcelink(){
   local src=${1:-$(env-home)/}
   svn-
   echo env:source:/trunk/$(env-rel $src)@$(svn-lastrev $src)
}

env-sourcetag(){ echo C2R ; }

env-designated(){ 
  [ -n "$ENV_DESIGNATED" ] && echo $ENV_DESIGNATED && return 0
  ## the below are for test servers 
  case ${1:-$NODE_TAG} in
    YY) echo YY ;;
    ZZ) echo ZZ ;;
     *) echo $(env-sourcetag) ;;
  esac
}

env-urlwc(){ svn info $1 | perl -n -e 'm,URL: (\S*), && print $1 ' -  ; }


env-abort-path(){ echo $(env-home)/ABORT ; }
env-abort-touch(){ touch $(env-abort-path) ; }
env-abort-clear(){ rm -f $(env-abort-path) ; }
env-abort(){
   local msg="=== $FUNCNAME :" 
   env-abort-touch
   echo $msg ABORT ... sleeping forever 
   sleep 1000000000000000 
}
env-abort-active-(){ [ -f "$(env-abort-path)" ] && return 0 || return 1  ;  }
env-abort-active(){ $FUNCNAME- && echo y || echo n ; }



env-relocate(){
   local msg="=== $FUNCNAME :"
   local tag=${1:-$(env-designated)}
   local reps="heprez env tracdev data aberdeen"
   local rep
   for rep in $reps ; do
   local wcd
     local wcd=$HOME/$rep
     if [ ! -d "$wcd" ] ; then
        echo -n
     elif [ ! -d "$wcd/.svn" ] ; then
        echo $msg skipping $wcd as not working copy 
     else
        env-relocate- $tag $wcd
     fi
   done
}


env-relocate-(){
  local msg="=== $FUNCNAME :"
  local tag=$1
  local wcd=${2:-$(env-home)}

  local url=$(env-url $tag $(basename $wcd))
  local urlwc=$(env-urlwc $wcd)
  local dtag=$(env-designated)

  [ "$tag" != "$dtag" ]  && echo $msg WARNING the designated tag $dtag differes from what you are relocating to 
  [ "$url" == "$urlwc" ] && echo $msg url of $wcd working copy  $urlwc  already matches that of tag $tag ... skipping && return 0 

  local iwd=$PWD
  cd $wcd
  local cmd="svn switch --relocate $urlwc $url "
  echo $msg from $PWD ... $cmd 
  local ans
  read -p "Switch source repository URL for WC ? enter YES to proceed "  ans
  [ "$ans" != "YES" ] && echo $msg skipping ... && return 0

  echo $msg proceeding...
  eval $cmd  

  cd $iwd

}


env-ihep(){  echo http://$1.ihep.ac.cn ; }
env-ntu(){   echo http://$1.phys.ntu.edu.tw ; }

env-localserver(){ 
  case ${1:-$(env-designated)} in 
     G) echo http://localhost ;;
     P) echo $(env-ntu grid1):8080 ;;
     C) echo $(env-ntu cms01) ;;
C2|C2R) echo $(env-ntu dayabay) ;;
    XX) echo $(env-ihep dyb2) ;;
    YY) echo $(env-ihep dyb1) ;;
    ZZ) echo $(env-ihep dayabay) ;;
  esac  
}

env-url(){         echo $(env-localserver $1)/repos/${2:-env}/trunk ; }
env-wikiurl(){     echo $(env-localserver $1)/tracs/${2:-env}/wiki ; }


env-email(){       echo blyth@hep1.phys.ntu.edu.tw ; }

setuptools-(){  . $(env-home)/python/setuptools.bash && setuptools-env $* ; }
sqlalchemy-(){  . $(env-home)/python/sqlalchemy.bash && sqlalchemy-env $* ; }
virtualenv-(){  . $(env-home)/python/virtualenv.bash && virtualenv-env $* ; }

mysql-(){       . $(env-home)/mysql/mysql.bash    && mysql-env $* ; }
log-(){         . $(env-home)/log/log.bash        && log-env $* ; }
phpbb-(){       . $(env-home)/phpbb/phpbb.bash    && phpbb-env $* ; }
etc-(){         . $(env-home)/base/etc.bash       && etc-env $* ; }
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
func-(){        . $(env-home)/base/func.bash    && func-env $* ; }
xmldiff-(){     . $(env-home)/xml/xmldiff.bash && xmldiff-env $* ; }

dyw-(){         . $(env-home)/dyw/dyw.bash   && dyw-env   $* ; }
oroot-(){       . $(env-home)/dyw/root.bash  && root-env  $* ; }
root-(){        . $(env-home)/root/root.bash  && root-env  $* ; }

_dyb__(){       . $(env-home)/dyb/dyb__.sh              $* ; }
dyb-(){         . $(env-home)/dyb/dyb.bash  && dyb-env  $* ; }
dybi-(){        . $(env-home)/dyb/dybi.bash && dybi-env $* ; }
dybr-(){        . $(env-home)/dyb/dybr.bash && dybr-env $* ; }
dybt-(){        . $(env-home)/dyb/dybt.bash && dybt-env $* ; }
dybpy-(){       . $(env-home)/dybpy/dybpy.bash && dybpy-env $* ; }
dybsvn-(){      . $(env-home)/dyb/dybsvn.bash && dybsvn-env $* ; }

abd-(){         . $(env-home)/aberdeen/abd.bash && abd-env $* ; }

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
htdocs-(){      . $(env-home)/trac/htdocs.bash && htdocs-env $* ; } 
tracpreq-(){    . $(env-home)/trac/tracpreq.bash && tracpreq-env $* ; } 
tmacros-(){     . $(env-home)/trac/macros/macros.bash  && tmacros-env $* ; }

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
 
 
  cd $iwd
  
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
  local iwd=$(pwd)
  cd $(env-home) 
  echo ============= env-u : status before update ================
  svn status -u
  svn update
  echo ============= env-u : status after update ================
  svn status -u
  cd $iwd
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
  local q=${1:-dummy}
  cd $(env-home)
  find . -name '*.*' -exec grep -H $q {} \;  | grep -v /.svn
}

env-dfind(){
  local q=${1:-dummy}

  if [ "$(uname)" == "Darwin" ]; then
     cd $(env-home)
     mdfind -onlyin . $q
  else
     env-find $*
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

env-notpath-(){ echo $PATH | grep -v $1 - > /dev/null  ; }
env-inpath-(){  echo $PATH | grep    $1 - > /dev/null  ; }

env-remove(){     export PATH=$(echo $PATH | perl -p -e "s,$1:,," - ) ; }
env-llp-remove(){ export $(env-libvar)=$(echo $PATH | perl -p -e "s,$1:,," - ) ; }

env-prepend(){
  local add=$1 
  env-notpath- $add && export PATH=$add:$PATH 
}

env-append(){
  local add=$1 
  env-notpath- $add && export PATH=$PATH:$add 
}

env-libvar(){
   case $(uname) in
      Darwin) echo DYLD_LIBRARY_PATH ;;
           *) echo LD_LIBRARY_PATH ;;
   esac
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

env-curl(){
  local msg="=== $FUNCNAME "
  local url=$1
  local cmd="curl -O $url "
  echo $msg $cmd $PWD
  eval $cmd
  [ "$?" != "0" ] && echo $msg FAILED : $cmd $PWD : SLEEPING && sleep 10000000000000  
}

env-mcurl(){
  local msg="=== $FUNCNAME "
  local url
  for url in $* ; do
     local cmd="curl -O $url "
     echo $msg $cmd $PWD
     eval $cmd
     [ "$?" != "0" ] && echo $msg FAILED : $cmd $PWD ...
  done 
  local name=$(basename $1)
  [ ! -f "$name" ] && echo $msg FAILED TO GET $name FROM ANY URL : SLEEPING && sleep 100000000000
}



env-ab(){

  local msg="=== $FUNCNAME :"
  apache-
  local tags="P C C2"
  local tag
  for tag in $tags ; do 
     local url=$(env-wikiurl $tag)
     local cmd="ab -v 2 -n 10 $url "   
     echo $msg tag $tag ... $cmd
     eval $cmd
  done

}

env-columns(){
   
   local msg="=== $FUNCNAME :"
   type $FUNCNAME
   echo COLUMNS : $COLUMNS
   echo tput cols : $(tput cols)
   python -c "import os ; os.environ['COLUMNS']='100' ; print os.environ['COLUMNS'] ; print os.getenv('COLUMNS')  "

}

env-htdocs-up(){
   local msg="=== $FUNCNAME :"
   local path=$1
   [ ! -f "$path" ] && echo $msg ABORT no path $path && return 1
   htdocs-
   TRAC_INSTANCE=env htdocs-up $path $(htdocs-privat)
}

env-htdocs-url(){
   local path=$1
   local name=$(basename $path)
   htdocs-
   echo $(TRAC_INSTANCE=env htdocs-url)/$(htdocs-privat)/$name
} 


env-override(){
   local msg="=== $FUNCNAME :"
   local path=$(env-override-path)
   if [ -f "$path" ] ; then
     source $path
   fi
}

env-override



#env-env
diff-(){       . $(env-home)/base/diff.bash && diff-env $* ; }
offdb-(){      . $(env-home)/offline/offdb.bash && offdb-env $* ; }
tg-(){         . $(env-home)/offline/tg/tg.bash && tg-env $*  ; }
pl-(){         . $(env-home)/offline/pl/pl.bash && pl-env $* ; }
pymysql-(){    . $(env-home)/db/pymysql.bash && pymysql-env $*  ; }
modwsgi-(){    . $(env-home)/apache/apachebuild/modwsgi.bash && modwsgi-env $* ; }
cpp-(){        . $(env-home)/base/cpp.bash && cpp-env $* ; } 
lighttpd-(){   . $(env-home)/lighttpd/lighttpd.bash && lighttpd-env $* ; }
pkgr-(){       . $(env-home)/base/pkgr.bash && pkgr-env $* ; }
modroot-(){    . $(env-home)/apache/modroot.bash && modroot-env $* ; }
tsocks-(){     . $(env-home)/proxy/tsocks.bash && tsocks-env $* ; }
pid-(){      . $(env-home)/base/pid.bash && pid-env $* ; }




rootd-(){      . $(env-home)/root/rootd.bash && rootd-env $* ; }
gallery3-(){      . $(env-home)/gallery3/gallery3.bash && gallery3-env $* ; }
nginx-(){      . $(env-home)/nginx/nginx.bash && nginx-env $* ; }
cpg-(){      . $(env-home)/cpg/cpg.bash && cpg-env $* ; }

dj-(){         . $(env-home)/dj/dj.bash && dj-env $*  ; }
djsa-(){      . $(env-home)/dj/djsa.bash && djsa-env $* ; }

djextensions-(){    . $(env-home)/dj/djextensions.bash && djextensions-env $* ; }
nosedjango-(){      . $(env-home)/dj/nosedjango.bash && nosedjango-env $* ; }
git-(){             . $(env-home)/git/git.bash && git-env $* ; }
formalchemy-(){     . $(env-home)/sa/formalchemy.bash && formalchemy-env $* ; }
rum-(){             . $(env-home)/rum/rum.bash && rum-env $* ; }
rumdev-(){          . $(env-home)/rum/rumdev.bash && rumdev-env $* ; }
twdev-(){      . $(env-home)/tw/twdev.bash && twdev-env $* ; }
