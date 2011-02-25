
# dummy minor commit to test bitten/queue

env-home(){     [ -n "$BASH_SOURCE" ] && [ "${BASH_SOURCE:0:1}" != "." ] &&  echo $(dirname $BASH_SOURCE) || echo $ENV_HOME ; }
env-source(){   echo $(env-home)/env.bash ; }
env-cd(){   cd $(env-home) ; }
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

env-gen(){ func-;func-gen $*;}

env-mode(){   echo dbg ; }
env-modedir(){  echo $(env-home)/scons-out/$(env-mode) ; }
env-testsdir(){ echo $(env-modedir)/tests ; }
env-bindir(){   echo $(env-modedir)/bin ; }
env-objdir(){   echo $(env-modedir)/obj ; }
env-libdir(){   echo $(env-modedir)/lib ; }
env-libpath(){
      [ "$1" == "blank" ] && echo -n && return 
      root-
      python-
      case $(hostname -s) in 
         cms01|cms02) echo $(env-libdir):$(root-libdir):$(python-libdir) ;;
          simon|g4pb) echo $(env-libdir):$(root-libdir):$(python-libdir) ;;
                   *) echo $(env-libdir):$(root-libdir):$(python-libdir) ;;
      esac
}

env-runenv-(){
      case $(uname) in
         Darwin) echo DYLD_LIBRARY_PATH=$(env-libpath $1) ;;
              *) echo LD_LIBRARY_PATH=$(env-libpath $1)  ;;
      esac
}
env-runpath-(){
   local xdi=$(dirname $(which xdpyinfo 2>/dev/null))
   local cat=$(dirname $(which cat 2>/dev/null))
   local ldd=$(dirname $(which ldd 2>/dev/null))
   local ipy=$(dirname $(which ipython 2>/dev/null))
   local gdb=$(dirname $(which gdb 2>/dev/null))
   echo $xdi:$cat:$ldd:$ipy:$gdb
}

 
env-runenv(){ cat << EOC
env -i $(env-runenv-) PATH=$(env-runpath-) DISPLAY=$DISPLAY PYTHONPATH=$ROOTSYS/lib ENV_HOME=$ENV_HOME ENV_PRIVATE_PATH=$ENV_PRIVATE_PATH HOME=$HOME USER=$USER ABERDEEN_HOME=$ABERDEEN_HOME
EOC
}


env-scons-(){ find $(env-home) -name '*.scons' ; }
env-scons(){ vi `$FUNCNAME-` ; }

env-config(){  $(env-home)/bin/env-config $* ; }
env-prefix(){ echo $(local-base)/env ; }

env-build(){

  python-
  python-ln $(env-home) env

  env-selinux
}

env-selinux(){

  apache-
  apache-chcon $(env-home)

  private-
  private-selinux
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
private-(){     . $(env-home)/private/private.bash && private-env $* ; }
func-(){        . $(env-home)/base/func.bash    && func-env $* ; }
xmldiff-(){     . $(env-home)/xml/xmldiff.bash && xmldiff-env $* ; }

dyw-(){         . $(env-home)/dyw/dyw.bash   && dyw-env   $* ; }
oroot-(){       . $(env-home)/dyw/root.bash  && root-env  $* ; }
root-(){        . $(env-home)/root/root.bash  && root-env  $* ; }

_dyb__(){       . $(env-home)/dyb/dyb__.sh              $* ; }
#dyb-(){         . $(env-home)/dyb/dyb.bash  && dyb-env  $* ; }
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
  local msg="=== $FUNCNAME :"
  local cmd="svn update $(env-home)"
  echo $msg \"$cmd\"
  eval $cmd
  [ -r $(env-home)/env.bash ] && . $(env-home)/env.bash  
  env-

  cd $(env-home)
  sct-
  sct

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
pl-(){         . $(env-home)/pl/pl.bash && pl-env $* ; }
pymysql-(){    . $(env-home)/mysql/pymysql.bash && pymysql-env $*  ; }
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

djext-(){    . $(env-home)/dj/djext.bash && djext-env $* ; }
nosedjango-(){      . $(env-home)/dj/nosedjango.bash && nosedjango-env $* ; }
git-(){             . $(env-home)/git/git.bash && git-env $* ; }
formalchemy-(){     . $(env-home)/sa/formalchemy.bash && formalchemy-env $* ; }
#rum-(){             . $(env-home)/rum/rum.bash && rum-env $* ; }
vip-(){             . $(env-home)/vip/vip.bash && vip-env $* ; }
rumdev-(){          . $(env-home)/rum/rumdev.bash && rumdev-env $* ; }
twdev-(){      . $(env-home)/tw/twdev.bash && twdev-env $* ; }
plvdbi-(){      . $(env-home)/plvdbi/plvdbi.bash && plvdbi-env $* ; }
vdbi-(){      . $(env-home)/vdbi/vdbi.bash && vdbi-env $* ; }
gitorious-(){      . $(env-home)/git/gitorious.bash && gitorious-env $* ; }
gitweb-(){      . $(env-home)/git/gitweb.bash && gitweb-env $* ; }
hgweb-(){      . $(env-home)/hg/hgweb.bash && hgweb-env $* ; }
authkit-(){      . $(env-home)/authkit/authkit.bash && authkit-env $* ; }
authkitdev-(){      . $(env-home)/authkit/authkitdev.bash && authkitdev-env $* ; }
hg-(){      . $(env-home)/hg/hg.bash && hg-env $* ; }
modscgi-(){      . $(env-home)/apache/apachebuild/modscgi.bash && modscgi-env $* ; }
modxsendfile-(){ . $(env-home)/apache/apachebuild/modxsendfile.bash && modxsendfile-env $* ; }
sv-(){      . $(env-home)/base/sv.bash && sv-env $* ; }
djdev-(){      . $(env-home)/dj/djdev.bash && djdev-env $* ; }
dybsite-(){      . $(env-home)/dj/dybsite/dybsite.bash && dybsite-env $* ; }
djdep-(){      . $(env-home)/dj/djdep.bash && djdep-env $* ; }
runinfo-(){      . $(env-home)/aberdeen/runinfo/runinfo.bash && runinfo-env $* ; }
svman-(){      . $(env-home)/base/svman.bash && svman-env $* ; }
svdev-(){      . $(env-home)/base/svdev.bash && svdev-env $* ; }
cfp-(){        . $(env-home)/base/cfp.bash && cfp-env $* ; }
pldep-(){      . $(env-home)/pl/pldep.bash && pldep-env $* ; }
plbook-(){      . $(env-home)/pl/plbook.bash && plbook-env $* ; }
ini-(){      . $(env-home)/base/ini.bash && ini-env $* ; }
jqplot-(){      . $(env-home)/jqplot/jqplot.bash && jqplot-env $* ; }
html-(){      . $(env-home)/base/html.bash && html-env $* ; }
redhat-(){      . $(env-home)/base/redhat.bash && redhat-env $* ; }
rabbitmq-(){      . $(env-home)/messaging/rabbitmq.bash && rabbitmq-env $* ; }
ahkera-(){      . $(env-home)/messaging/ahkera.bash && ahkera-env $* ; }
openamq-(){      . $(env-home)/messaging/openamq.bash && openamq-env $* ; }
selinux-(){      . $(env-home)/base/selinux.bash && selinux-env $* ; }
carrot-(){      . $(env-home)/messaging/carrot/carrot.bash && carrot-env $* ; }
pcre-(){      . $(env-home)/pcre/pcre.bash && pcre-env $* ; }
cjson-(){      . $(env-home)/messaging/cjson.bash && cjson-env $* ; }
rootmq-(){      . $(env-home)/rootmq/rootmq.bash && rootmq-env $* ; }
alice-(){      . $(env-home)/messaging/alice.bash && alice-env $* ; }
priv-(){      . $(env-home)/priv/priv.bash && priv-env $* ; }
bunny-(){      . $(env-home)/messaging/bunny.bash && bunny-env $* ; }
html-(){      . $(env-home)/html/html.bash && html-env $* ; }
rootana-(){      . $(env-home)/root/rootana.bash && rootana-env $* ; }
#daily-(){      . $(env-home)/nuwa/daily.bash && daily-env $* ; }  THIS HAS MOVED INTO DYBSVN
db-(){      . $(env-home)/db/db.bash && db-env $* ; }
djnose-(){      . $(env-home)/dj/djnose.bash && djnose-env $* ; }
memoir-(){      . $(env-home)/latex/memoir.bash && memoir-env $* ; }
pinocchio-(){      . $(env-home)/nose/package/pinocchio.bash && pinocchio-env $* ; }
scons-(){      . $(env-home)/scons/scons.bash && scons-env $* ; }
minini-(){      . $(env-home)/base/minini.bash && minini-env $* ; }
lxr-(){      . $(env-home)/doc/lxr.bash && lxr-env $* ; }
doxygen-(){      . $(env-home)/doc/doxygen.bash && doxygen-env $* ; }
bitn-(){      . $(env-home)/trac/slave/bitn.bash && bitn-env $* ; }
tracdoxygen-(){      . $(env-home)/trac/package/tracdoxygen.bash && tracdoxygen-env $* ; }
abtviz-(){      . $(env-home)/AbtViz/abtviz.bash && abtviz-env $* ; }  
tracdep-(){      . $(env-home)/trac/tracdep.bash && tracdep-env $* ; }
jmeter-(){      . $(env-home)/http/jmeter.bash && jmeter-env $* ; }
scube-(){      . $(env-home)/scons/scube.bash && scube-env $* ; }
sct-(){        . $(env-home)/scons/sct.bash && sct-env $* ; }
parts-(){      . $(env-home)/scons/parts.bash && parts-env $* ; }
omaha-(){      . $(env-home)/scons/omaha.bash && omaha-env $* ; }
rmqc-(){       . $(env-home)/messaging/rmqc.bash && rmqc-env $* ; }
cjsn-(){      . $(env-home)/cjsn/cjsn.bash && cjsn-env $* ; }
pkgconfig-(){      . $(env-home)/base/pkgconfig.bash && pkgconfig-env $* ; }
sqlite3x-(){      . $(env-home)/db/sqlite3x.bash && sqlite3x-env $* ; }
sq3-(){      . $(env-home)/db/sq3.bash && sq3-env $* ; }
slv-(){      . $(env-home)/trac/slave/slv.bash && slv-env $* ; }
ejabberd-(){      . $(env-home)/messaging/ejabberd.bash && ejabberd-env $* ; }
modrabbitmq-(){      . $(env-home)/messaging/modrabbitmq.bash && modrabbitmq-env $* ; }
adium-(){      . $(env-home)/messaging/adium.bash && adium-env $* ; }
pika-(){      . $(env-home)/messaging/pika.bash && pika-env $* ; }
glib-(){      . $(env-home)/base/glib.bash && glib-env $* ; }
erlang-(){      . $(env-home)/erlang/erlang.bash && erlang-env $* ; }
strophe-(){      . $(env-home)/messaging/strophe.bash && strophe-env $* ; }
speeqe-(){      . $(env-home)/messaging/speeqe.bash && speeqe-env $* ; }
erl-(){      . $(env-home)/erlang/erl.bash && erl-env $* ; }
pyxmpp-(){      . $(env-home)/messaging/pyxmpp.bash && pyxmpp-env $* ; }
xmpppy-(){      . $(env-home)/messaging/xmpppy.bash && xmpppy-env $* ; }

hash-(){      . $(env-home)/base/hash.bash && hash-env $* ; }
nenv-(){      . $(env-home)/nuwa/nenv.bash && nenv-env $* ; }
dyb-(){      . $(env-home)/nuwa/dyb.bash && dyb-env $* ; }
du-(){      . $(env-home)/base/du.bash && du-env $* ; }
gendbi-(){      . $(env-home)/offline/gendbi/gendbi.bash && gendbi-env $* ; }
time-(){      . $(env-home)/python/tests/time.bash && time-env $* ; }
svlisten-(){      . $(env-home)/sysadmin/sv/svlisten.bash && svlisten-env $* ; }
diskmon-(){      . $(env-home)/sysadmin/sv/diskmon.bash && diskmon-env $* ; }
svnauthzadmin-(){      . $(env-home)/trac/package/svnauthzadmin.bash && svnauthzadmin-env $* ; }
guardian-(){      . $(env-home)/dj/guardian.bash && guardian-env $* ; }
dybprj-(){      . $(env-home)/dybprj/dybprj.bash && dybprj-env $* ; }
nodejs-(){      . $(env-home)/nodejs/nodejs.bash && nodejs-env $* ; }
rabbitjs-(){      . $(env-home)/nodejs/rabbitjs.bash && rabbitjs-env $* ; }
nodeamqp-(){      . $(env-home)/nodejs/nodeamqp.bash && nodeamqp-env $* ; }
socketio-(){      . $(env-home)/nodejs/socketio.bash && socketio-env $* ; }
djylt-(){      . $(env-home)/dj/djylt.bash && djylt-env $* ; }
djcelery-(){      . $(env-home)/dj/djcelery.bash && djcelery-env $* ; }
telnet-(){      . $(env-home)/base/telnet.bash && telnet-env $* ; }
matplotlib-(){      . $(env-home)/matplotlib/matplotlib.bash && matplotlib-env $* ; }
timeseries-(){      . $(env-home)/scikits/timeseries.bash && timeseries-env $* ; }
numpy-(){      . $(env-home)/npy/numpy.bash && numpy-env $* ; }
cython-(){      . $(env-home)/npy/cython.bash && cython-env $* ; }
mysql-pyrex-(){      . $(env-home)/db/mysql-pyrex.bash && mysql-pyrex-env $* ; }
mysql-python-(){      . $(env-home)/mysql/mysql-python.bash && mysql-python-env $* ; }
camqadm-(){      . $(env-home)/messaging/camqadm.bash && camqadm-env $* ; }
celery-(){      . $(env-home)/messaging/celery.bash && celery-env $* ; }
pygtk-(){      . $(env-home)/gui/pygtk.bash && pygtk-env $* ; }
pygobject-(){      . $(env-home)/gui/pygobject.bash && pygobject-env $* ; }
mplh5canvas-(){      . $(env-home)/matplotlib/mplh5canvas.bash && mplh5canvas-env $* ; }
gnocl-(){      . $(env-home)/gui/gnocl.bash && gnocl-env $* ; }
sphinx-(){      . $(env-home)/doc/sphinx.bash && sphinx-env $* ; }
converter-(){      . $(env-home)/doc/converter.bash && converter-env $* ; }
redmine-(){      . $(env-home)/redmine/redmine.bash && redmine-env $* ; }
redmine-hudson-(){      . $(env-home)/redmine/redmine-hudson.bash && redmine-hudson-env $* ; }
hudson-(){      . $(env-home)/hudson/hudson.bash && hudson-env $* ; }
buildbot-(){      . $(env-home)/buildbot/buildbot.bash && buildbot-env $* ; }
scipy-(){      . $(env-home)/npy/scipy.bash && scipy-env $* ; }
analog-(){      . $(env-home)/apache/analog.bash && analog-env $* ; }
tornado-(){      . $(env-home)/messaging/tornado.bash && tornado-env $* ; }
