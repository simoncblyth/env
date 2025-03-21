[ -e ~/env.local.bash ] && . ~/env.local.bash 

env-logpath(){ echo $(env-home)/docs/log/$(date +"%b%Y").rst ; }
env-logpath(){ echo $(env-home)/docs/log/$(date +"%b%Y").rst ; }
env-log(){ vi $(${FUNCNAME}path) ; }

env-commit-url(){ echo https://bitbucket.org/simoncblyth/env/commits/${1:-658f4429167b} ; }
env-commit(){ open $(env-commit-url $1); }


env-src-url(){ echo https://bitbucket.org/simoncblyth/env/src/tip/${1:-chroma/G4DAEChroma} ; }
env-src(){ open $(env-src-url $1); }



env-cache-path(){ echo ~/env-cache.sh ; }
env-cache(){
   local path=$(env-cache-path)
   [ ! -f "$path" ] && env-cache-update
   echo $msg sourcing env from $path
   source $path 
}
env-cache-update(){
   local path=$(env-cache-path)
   local msg="=== $FUNCNAME :"
   echo $msg writing env to $path
   env | sort | grep -v SSH_ | grep -v DISPLAY | grep -v TERM | grep -v LS_COLORS | grep -v PWD | grep -v DAE_PATH_TEMPLATE | while read line ; do echo export $line ; done > $path
}

env-cls(){
    local base="."
    local name=${1:-G4Scintillation}
    local h=$(find $base -name "$name.h")
    local hh=$(find $base -name "$name.hh")
    local cc=$(find $base -name "$name.cc")
    local cxx=$(find $base -name "$name.cxx")
    local icc=$(find $base -name "$name.icc")
    local vcmd="vi -R $h $hh $icc $cc $cxx"
    echo $vcmd
    eval $vcmd
}

env-f(){ find . -not -iwholename '*.svn*' -a -type f  ; }
env-q(){ find . -not -iwholename '*.svn*' -a -type f -exec grep -H ${1:-Sensitive} {} \; ; }


env-touch-index(){
   # recursivly touch existing index.rst going up the directory tree from PWD
   # for pursuading sphinx to rebuild a deep page
   local dir=${1:-$PWD}
   [ "$dir" == "/" ] && return 
   local idx=$dir/index.rst
   [ -f "$idx" ] && echo $idx && touch $idx 
   $FUNCNAME $(dirname $dir)
}


env-clean-build()
{
   cd $LOCAL_BASE/env

   local msg="=== $FUNCNAME $PWD :"
   find . -type d -name '*.build'  -exec du -hs {} \;

   local ans
   read -p "$msg enter YES to proceed with deleting the above .build folders : " ans
   [ "$ans" != "YES" ] && echo $msg skipped && return 

   local name
   find . -type d -name '*.build' | while read name ; do
      echo \"$name\"
      rm -rf "$name"
   done

}


env-anewer(){
   cd $(env-home)
   local path=${1:-graphics/openscenegraph/osg.bash}
   find . -anewer $path  -type f | grep -v _build | grep -v .svn
}

env-htdocs-rsync(){
   local target=${1:-C2}
   local msg="=== $FUNCNAME :"
   [ "$target" == "$NODE_TAG" ] && echo $msg ABORT : CANNOT RSYNC TO SELF && return 1

   apache-
   local cmd="rsync -avz --exclude '*.mov' $(apache-htdocs)/env/ $target:$(apache-htdocs $target)/env/"
   echo $msg $cmd
   eval $cmd
}


env_u_(){ 
   # mapping directory to url  : hmmm this should live in home- not here ??
   local dir=${1:-$PWD}
   local abs=$(realpath $dir)   # see ~/e/tools/realpath/

   local envhome=${ENV_HOME:-$HOME/env} 
   local workflowhome=${WORKFLOW_HOME:-$HOME/workflow} 
   local homehome=${HOME_HOME:-$HOME/home} 
   local optickshome=${OPTICKS_HOME:-$HOME/opticks} 

   case $abs in
           ${envhome}) echo http://localhost/env_notes/ ;; 
          ${envhome}*) echo http://localhost/env_notes/${abs/$envhome\/}/ ;; 
      ${workflowhome}) echo http://localhost/w/ ;; 
     ${workflowhome}*) echo http://localhost/w/${abs/$workflowhome\/}/ ;; 
          ${homehome}) echo http://localhost/h/ ;; 
         ${homehome}*) echo http://localhost/h/${abs/$homehome\/}/ ;; 
       ${optickshome}) echo https://bitbucket.org/simoncblyth/opticks/src/master/ ;; 
      ${optickshome}*) echo https://bitbucket.org/simoncblyth/opticks/src/master/${abs/$optickshome\/}/ ;; 
                    *) echo http://www.google.com ;;
   esac
}
u(){ 
   local url=$(env_u_ $*)
   echo $msg open URL corresponding to PWD $PWD : $url
   [ "$(uname)" == "Darwin" ] && open $url 
}

env_tip_(){
   local dir=${1:-$PWD}
   local abs=$(realpath $dir)   # see ~/e/tools/realpath/

   ## TODO: detect hg or git and set cloud url according to clone metadata
   case $abs in
       ${ENV_HOME}) echo https://bitbucket.org/simoncblyth/env/src/tip/ ;;
      ${ENV_HOME}*) echo https://bitbucket.org/simoncblyth/env/src/tip/${abs/$ENV_HOME\/}/ ;;
                 *) echo http://www.google.com ;;
   esac 
}

tip(){
   local url=$(env_tip_ $*)
   echo $msg open URL corresponding to PWD $PWD : $url
   [ "$(uname)" == "Darwin" ] && open $url 
}





env-open(){
  local url=${1:-stackoverflow.com}
  if [ "$(which open 2> /dev/null)" == "" ]; then 
      clui-
      clui-open $url
  else
      open $url
  fi

}



env-usage(){ cat << EOU

===================
ENV BASH FUNCTIONS
===================

Bash function reminders
------------------------

type name        
    list a function definition 
set               
    list all functions
unset -f name     
    remove a function
typeset -F        
    lists just the names
local 
    from inside a function lists the locals, eg::
    
        ff(){ local a="hello" ; local b="world" ; local ; }  
	ff

* http://www.network-theory.co.uk/docs/bashref/ShellFunctions.html
* http://www-128.ibm.com/developerworks/library/l-bash-test.html

bad workaround blockade of cms02 from belle7
----------------------------------------------

On G::

   svn up ~/env ; svn export --force ~/env ~/eenv
   cd ; rsync -av eenv N:                ## deletes do not get propagated with this

On N::

   export  ENV_HOME=/home/blyth/eenv ; ## modify in .bash_profile to use the exported

better workaround via reverse proxy on H 
------------------------------------------

#. requies nginx to be running on H 
#. one time `svn switch --relocate` is needed to keep working with the env WC on N::

	[blyth@belle7 e]$ svn switch --relocate http://dayabay.phys.ntu.edu.tw/repos/env http://hfag.phys.ntu.edu.tw:90/repos/env

#. ditto for heprez::
        [blyth@belle7 heprez]$ svn switch --relocate http://dayabay.phys.ntu.edu.tw/repos/heprez http://hfag.phys.ntu.edu.tw:90/repos/heprez   



switch back now that the block is removed
-------------------------------------------
::

    svn switch --relocate http://hfag.phys.ntu.edu.tw:90/repos/env http://dayabay.phys.ntu.edu.tw/repos/env



env functions
---------------

*env-dbg*
     invoke with bash rather than . when debugging to see 
     line numbers of errors, CAUTION error reporting can be a line off

*env-rsync top-level-fold [target-node]*
     propagate a top-level-folder without svn, caution can
     leave SVN wc state awry ... usually easiest to delete working
     copy and "svn up" when want to come clean and go back to SVN
     
*env-rsync-all   [target-node]*
     rsync env working copy excluding .svn etc..
     to a list of remote nodes specified by ssh node tag 

*env-again*
     delete working copy and checkout again 

*env-u*
     update the working copy, aliased to "eu" 
          
*env-rst name.bash*
     create rst wrapper to include the usage bash function

*env-toc*
     create index.rst toctree referencing all .rst in PWD 

*env-wdir dir:-PWD*
     create index.rst using wdir directive to provide a list of .pdf 
     used for parallel or sweep.py trees  


absorbing an exported env working copy 
-----------------------------------------

Useful following server downtime::

     [blyth@cms02 ~]$ diff -r --brief env env.keep | grep -v .svn | grep differ | perl -p -e 's,Files (\S*) and (\S*) differ,cp $2 $1, ' -


toctree hookup
---------------

When referring to implicit indices such as "python/python.bash" or "tools/tools.bash" use the 
form "python/index". The bash2rst tool does the appropriate path swapping to generated a
cleaner output tree without repeated levels.

.. warning:: this could be confusing when using manual index.rst together with auto generated *_docs* 


======
FUNCS
======

.. toctree::

    python/index
    tools/index



EOU
}

env-home(){     [ -n "$BASH_SOURCE" ] && [ "${BASH_SOURCE:0:1}" != "." ] &&  echo $(dirname $BASH_SOURCE) || echo $ENV_HOME ; }
env-source(){   echo $(env-home)/env.bash ; }
env-cd(){   cd $(env-home) ; }

env-pwdx(){
   # determine cwd of another process on OSX, on Linux just use pwdx 
   local pid=$1
   lsof -a -p $pid -d cwd -Fn | cut -c2- | grep -v $pid
}


env-rdir(){
  local home=$(env-home)
  local para=$(local-base)/env
  local here=$(pwd -P)  # physical with symlinks resolved
  case $here in 
     $home) echo -n ;;
     $para) echo -n ;;
    $home*) echo ${here/$home\/} ;;
    $para*) echo ${here/$para\/} ;;
         *) echo here $here is not inside home $home or para $para 1>&2 && echo .  ;; 
  esac 
}
env-pdir(){
  local home=$(env-home)
  local para=$(local-base)/env
  local here=$(pwd -P)  # physical with symlinks resolved
  case $here in 
     $home) echo $para ;;
     $para) echo $home ;;
    $home*) echo $para/${here/$home\/} ;;
    $para*) echo $home/${here/$para\/} ;;
         *) echo here $here is not inside home $home or para $para 1>&2 && echo .  ;; 
  esac 
}
env-para(){ cd $(env-pdir) ; }
pd(){ 
   : shortcut function from env/env.bash 
   env-para && pwd 
}

env-vi(){       vi $(env-source) ; }
env-ini(){      . $(env-source) ; }
env-check-svn() {
    [  ! -z $(env-lastrev) ] && return 0
    local msg="=== $FUNCNAME"
    echo $msg "SVN issue, check env-lastrev"
    return 1
}
env-lastrev(){  svn- ; svn-lastrev $(env-home) ; }
env-override-path(){ echo $(env-home)/local/$(uname -n).bash ; }
env-rel(){
  local src=$1
  local rel=${src/$(env-home)\//}
  [ "$rel" == "$src" ] && rel=$(env-home)/
  echo $rel
}

env-gen(){      func-;func-gen $*;}

env-genproj(){  proj-;proj-gen env $*;}
env-fgenproj(){ proj-;proj-fgen env $*;}

env-toc-(){ cat << EOX

$(basename $PWD)
=========================

.. toctree::

EOX
  ls -1 *.rst | grep -v index.rst | while read line ; do printf "   %s\n" ${line/.rst} ;  done 

echo ""
}

env-toc(){
   [ -f index.rst ] && echo index.rst exists already in PWD $PWD ----------- && cat index.rst && echo ------------------

   local ans
   read -p "Enter YES to delete this and recreate " ans
   [ "$ans" != "YES" ] && echo OK skipping && return 
   

   $FUNCNAME- > index.rst
}

env-rst-(){  
  cat << EOZ 

.. include:: $name
   :start-after: cat << EO$token
   :end-before: EO$token

EOZ
}
env-rst-xml-(){  
  cat << EOZ 

.. include:: $name
   :start-after: <description>
   :end-before: </description>

EOZ
}

env-rst(){
   local path=$1
   local token=${2:-U}   # have to avoid saying E O U together due to bash2rst 
   local dir=$(dirname $path)
   local name=$(basename $path)
   local base
   if [ "$name" == "build.xml" ]; then 
       base=${name/.xml}
   else
       base=${name/.bash}
   fi
   local rstpath=$dir/$base.rst
   [ -f "$rstpath" ] && echo $msg rst $rstpath exists already, delete and rerun to proceed && return 

   case $name in 
     build.xml) $FUNCNAME-xml- > $rstpath ;;
             *) $FUNCNAME-     > $rstpath ;;
   esac
}


env-wdir-(){ cat <<EOL
:wdir:

WDIR
=====

PDFS
----

.. wdir:: *.pdf

DOCX
-----

.. wdir:: *.docx


EOL
}

env-wdir(){

  local dir=${1:-$PWD} 
  local path=$dir/index.rst
  [ -f "$path" ] && echo $msg path $path exists already && return 

  echo $msg writing $path 
  $FUNCNAME- > $path
}


env-export(){
   local dir=/tmp/export/env
   mkdir -p $(dirname $dir)
   svn export $(env-home) $dir
}

env-docs(){
   cd $(env-home)
   python rstbash.py
}

env-sphinx(){
   python-
   local msg="=== $FUNCNAME"
   cmd="cd $(env-home) && PATH=$(env-home)/bin:$PATH make && make rsync "
   echo $msg $cmd updating html 
   eval $cmd 
}

env-mode(){   echo dbg ; }
env-modedir(){  echo $(env-home)/scons-out/$(env-mode) ; }
env-testsdir(){ echo $(env-modedir)/tests ; }
env-bindir(){   echo $(env-modedir)/bin ; }
env-objdir(){   echo $(env-modedir)/obj ; }
env-libdir(){   echo $(env-modedir)/lib ; }
env-libpath(){
      [ "$1" == "blank" ] && echo -n && return 
      eroot-
      python-
      case $(hostname -s) in 
         cms01|cms02) echo $(env-libdir):$(eroot-libdir):$(python-libdir) ;;
          simon|g4pb) echo $(env-libdir):$(eroot-libdir):$(python-libdir) ;;
                   *) echo $(env-libdir):$(eroot-libdir):$(python-libdir) ;;
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
    Y1) echo Y1 ;;
    Y2) echo Y2 ;;
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
    Y1) echo http://202.122.39.101 ;;
    Y2) echo $(env-ihep dayabay-new) ;;
    ZZ) echo $(env-ihep dayabay) ;;
  esac  
}

env-url(){         echo $(env-localserver $1)/repos/${2:-env}/trunk ; }
env-wikiurl(){     echo $(env-localserver $1)/tracs/${2:-env}/wiki ; }


env-email(){       echo blyth@hep1.phys.ntu.edu.tw ; }

setuptools-(){  . $(env-home)/python/setuptools.bash && setuptools-env $* ; }
sqlalchemy-(){  . $(env-home)/python/sqlalchemy.bash && sqlalchemy-env $* ; }
virtualenv-(){  . $(env-home)/python/virtualenv.bash && virtualenv-env $* ; }

pdsh-(){        . $(env-home)/python_course/pdsh.bash && pdsh-env $* ; }
pyc-(){         . $(env-home)/python_course/pyc.bash && pyc-env $* ; }

mysql-(){       . $(env-home)/mysql/mysql.bash    && mysql-env $* ; }
log-(){         . $(env-home)/log/log.bash        && log-env $* ; }
phpbb-(){       . $(env-home)/phpbb/phpbb.bash    && phpbb-env $* ; }
etc-(){         . $(env-home)/base/etc.bash       && etc-env $* ; }
cronline-(){    . $(env-home)/base/cronline.bash && cronline-env $* ; }
cvmfs-(){       . $(env-home)/base/cvmfs.bash    && cvmfs-env $* ; }
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
droot-(){       . $(env-home)/dyw/droot.bash  && eroot-env  $* ; }
eroot-(){       . $(env-home)/root/eroot.bash  && eroot-env  $* ; }

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


sqlite-(){      . $(env-home)/sqlite/sqlite.bash && sqlite-env $* ; } 

swish-(){       . $(env-home)/swish/swish.bash && swish-env $* ; } 

cvs-(){         . $(env-home)/cvs/cvs.bash && cvs-env $* ; } 



db-(){          . $(env-home)/db/db.bash     && db-env $*     ; }





pexpect-(){      . $ENV_HOME/python/pexpect.bash   && pexpect-env   $* ; }
configobj-(){    . $ENV_HOME/python/configobj.bash && configobj-env $* ; }
pythonbuild-(){  . $ENV_HOME/python/pythonbuild/pythonbuild.bash && pythonbuild-env $* ; }
python-(){      . $(env-home)/python/python.bash  && python-env $*  ; }
ipython-(){     . $(env-home)/python/ipython.bash && ipython-env $* ; }



cpprun-(){      . $(env-home)/cpprun/cpprun.bash && cpprun-env $* ; }
macros-(){      . $(env-home)/macros/macros.bash && macros-env $* ; }
offline-(){     . $(env-home)/offline/offline.bash && offline-env $* ; }


xml-(){         . $(env-home)/xml/xml.bash ; }





  
# the below may not work in non-interactive running ???  
md-(){  local f=${FUNCNAME/-} && local p=$(env-home)/$f/$f.bash && [ -r $p ] && . $p ; } 
 
 
ee(){ cd $(env-home)/$1 ; }
 
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
  #svn-
 
  PATH=$(env-home)/bin:$PATH
 
  cd $iwd
  
  #cmt-  
  
  #boost-
  #boost-export

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
  find . -name '*.*' -type f -exec grep -H $q {} \;  | grep -v /.svn
}
env-rstfind(){
  local q=${1:-dummy}
  cd $(env-home)
  find . -name '*.rst' -exec grep -H $q {} \;  
}
env-bashfind(){
  local q=${1:-dummy}
  cd $(env-home)
  find . -name '*.bash' -exec grep -H $q {} \;  
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


env-path(){   echo $PATH | tr ":" "\n" ; }
env-pathpl(){ echo $PATH | perl -pe 's,:,\n,g' - ;  }

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





env-index-head-(){ cat << EOH 
<html>
<head>
</head>
<body>
<table>
  <tr>
      <td> Trac </td>
      <td> SVN </td>
      <td> Sphinx </td>
  </tr>
EOH
}

env-index(){
  $FUNCNAME-head-
  $FUNCNAME-body-  /tracs/env/     /repos/env/trunk/       /edocs
  $FUNCNAME-body-  /tracs/heprez/  /repos/heprez/trunk/    /hdocs
  $FUNCNAME-body-  /tracs/tracdev/ /repos/tracdev/
  $FUNCNAME-tail-
}


env-index-body-(){ cat << EOB
  <tr>
      <td>  <a href="$1" > $1 </a> </td>
      <td>  <a href="$2" > $2 </a> </td>
      <td>  <a href="$3" > $3 </a> </td>
  </tr>
EOB
}

env-index-tail-(){ cat << EOT
</table>
</body>
</html>
EOT
}

# just using symbolic link works more simply 
#env-pth(){
#  local libdir=${1:-$(env-home)}
#  local libnam=${2:-$(basename $libdir)}
#  local msg="=== $FUNCNAME :"
#  python-
#  local site=$PYTHON_SITE
#  [ ! -d "$site" ] && echo $msg site $site does not exist && return 
#  local pth=$site/$libnam.pth
#  [ -f "$pth" ] && echo $msg pth $pth exists already && cat $pth && ls -l $pth && return 
#  echo $msg writing libnam $libnam to pth $pth 
#  sudo bash -c "echo $libdir  > $pth"
#}


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

chomp(){   perl -pi -e 'chomp if eof' $* ; : env/env.bash ;  }

djext-(){    . $(env-home)/dj/djext.bash && djext-env $* ; }
nosedjango-(){      . $(env-home)/dj/nosedjango.bash && nosedjango-env $* ; }
git-(){             . $(env-home)/git/git.bash && git-env $* ; }
gitu-(){            . $(env-home)/git/gitu.bash && gitu-env $* ; }
grep-(){            . $(env-home)/grep/grep.bash && grep-env $* ; }
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
svnprecommit-(){      . $(env-home)/svn/svnprecommit.bash && svnprecommit-env $* ; }
dcs-(){      . $(env-home)/offline/dcs.bash && dcs-env $* ; }
#scr-(){      . $(env-home)/offline/scr.bash && scr-env $* ; }
drush-(){      . $(env-home)/drush/drush.bash && drush-env $* ; }
videola-(){  . $(env-home)/videola/videola.bash && videola-env $* ; }
mediamosa-(){      . $(env-home)/mediamosa/mediamosa.bash && mediamosa-env $* ; }
tcpdump-(){      . $(env-home)/base/tcpdump.bash && tcpdump-env $* ; }
svnlog-(){      . $(env-home)/tools/svnlog.bash && svnlog-env $* ; }
linux-(){      . $(env-home)/tools/linux.bash && linux-env $* ; }
macports-(){      . $(env-home)/base/macports.bash && macports-env $* ; }
xsd-(){      . $(env-home)/xml/xsd.bash && xsd-env $* ; }
bdbxml-(){      . $(env-home)/db/bdbxml.bash && bdbxml-env $* ; }
swig-(){      . $(env-home)/swig/swig.bash && swig-env $* ; }
swigbuild-(){      . $(env-home)/swig/swigbuild.bash && swigbuild-env $* ; }
viruscheck-(){      . $(env-home)/admin/viruscheck.bash && viruscheck-env $* ; }
dbxml-(){      . $(env-home)/db/dbxml.bash && dbxml-env $* ; }
boostlog-(){      . $(env-home)/boost/boostlog.bash && boostlog-env $* ; }
edocutils-(){      . $(env-home)/python/edocutils.bash && edocutils-env $* ; }
fabric-(){      . $(env-home)/tools/fabric.bash && fabric-env $* ; }
cuisine-(){      . $(env-home)/tools/cuisine.bash && cuisine-env $* ; }
daemonwatch-(){      . $(env-home)/tools/daemonwatch.bash && daemonwatch-env $* ; }
highstock-(){      . $(env-home)/plot/highstock.bash && highstock-env $* ; }
highcharts-(){      . $(env-home)/plot/highcharts.bash && highcharts-env $* ; }
njs-(){      . $(env-home)/nodejs/njs.bash && njs-env $* ; }
nodehighcharts-(){      . $(env-home)/nodejs/nodehighcharts.bash && nodehighcharts-env $* ; }
sphinxcontrib-(){      . $(env-home)/doc/sphinxcontrib.bash && sphinxcontrib-env $* ; }
trace-(){      . $(env-home)/sysadmin/trace.bash && trace-env $* ; }
tools-(){      . $(env-home)/tools/tools.bash && tools-env $* ; }

tracxmlrpc-(){   . $(env-home)/otrac/tracxmlrpc.bash && tracxmlrpc-env $* ; }
trachttpauth-(){ . $(env-home)/otrac/trachttpauth.bash && trachttpauth-env $* ; }
trac2mediawiki-(){ . $(env-home)/trac/package/hide/trac2mediawiki.bash && trac2mediawiki-env $* ; }
tracwiki2sphinx-(){ . $(env-home)/trac/package/hide/tracwiki2sphinx.bash    && tracwiki2sphinx-env $* ; }

twiki-(){      . $(env-home)/twiki/twiki.bash && twiki-env $* ; }
ttocplugin-(){      . $(env-home)/twiki/ttocplugin.bash && ttocplugin-env $* ; }
getgitorious-(){      . $(env-home)/git/getgitorious.bash && getgitorious-env $* ; }
mysqludfsys-(){     . $(env-home)/db/mysqludfsys.bash && mysqludfsys-env $* ; }
jinja2-(){      . $(env-home)/tools/jinja2.bash && jinja2-env $* ; }
fossil-(){      . $(env-home)/fossil/fossil.bash && fossil-env $* ; }
envcap-(){      . $(env-home)/tools/envcap.bash && envcap-env $* ; }
cfg-(){         . $(env-home)/tools/cfg.bash && cfg-env $* ; }

f-(){      . $(env-home)/fossil/f.bash && f-env $* ; }
gitlab-(){      . $(env-home)/git/gitlab.bash && gitlab-env $* ; }
distribute-(){      . $(env-home)/python/distribute.bash && distribute-env $* ; }
find-(){      . $(env-home)/tools/find.bash && find-env $* ; }
psutil-(){      . $(env-home)/tools/psutil.bash && psutil-env $* ; }
mysqlrpm-(){      . $(env-home)/mysql/mysqlrpm.bash && mysqlrpm-env $* ; }
pysqlite-(){    . $(env-home)/sqlite/pysqlite.bash && pysqlite-env $* ; }


trac2bitbucket-(){      . $(env-home)/scm/bitbucket/trac2bitbucket.bash && trac2bitbucket-env $* ; }
proxy-(){      . $(env-home)/proxy/proxy.bash && proxy-env $* ; }
jni_hello_world-(){      . $(env-home)/java/jni/jni_hello_world.bash && jni_hello_world-env $* ; }
jexec-(){      . $(env-home)/java/commons/exec/jexec.bash && jexec-env $* ; }
launchctl-(){      . $(env-home)/osx/launchctl.bash && launchctl-env $* ; }
svgplotlib-(){      . $(env-home)/plot/svgplotlib/svgplotlib.bash && svgplotlib-env $* ; }
svgcharts-(){      . $(env-home)/plot/svgcharts/svgcharts.bash && svgcharts-env $* ; }
cqpack-(){      . $(env-home)/plot/svgcharts/cqpack.bash && cqpack-env $* ; }
svn2git-(){      . $(env-home)/scm/svn2git/svn2git.bash && svn2git-env $* ; }
opw-(){      . $(env-home)/muon_simulation/optical_photon_weighting/opw.bash && opw-env $* ; }
chroma-(){      . $(env-home)/chroma/chroma.bash && chroma-env $* ; }
fast-(){      . $(env-home)/tools/fast/fast.bash && fast-env $* ; }
gperftools-(){      . $(env-home)/tools/gperftools/gperftools.bash && gperftools-env $* ; }
kcachegrind-(){      . $(env-home)/tools/kcachegrind.bash && kcachegrind-env $* ; }
ocelot-(){      . $(env-home)/cuda/ocelot.bash && ocelot-env $* ; }
cudatoolkit-(){      . $(env-home)/cuda/cudatoolkit.bash && cudatoolkit-env $* ; }
gcc-(){      . $(env-home)/base/gcc.bash && gcc-env $* ; }
llvm-(){      . $(env-home)/llvm/llvm.bash && llvm-env $* ; }
meshlab-(){      . $(env-home)/graphics/meshlab/meshlab.bash && meshlab-env $* ; }
freewrl-(){      . $(env-home)/graphics/vrml/freewrl.bash && freewrl-env $* ; }
vrml97import-(){      . $(env-home)/graphics/vrml/vrml97import.bash && vrml97import-env $* ; }
blender-(){      . $(env-home)/graphics/blender/blender.bash && blender-env $* ; }
pyopencl-(){      . $(env-home)/opencl/pyopencl.bash && pyopencl-env $* ; }
meshpy-(){      . $(env-home)/graphics/mesh/meshpy.bash && meshpy-env $* ; }
vtk-(){      . $(env-home)/graphics/vtk/vtk.bash && vtk-env $* ; }
pycuda-(){      . $(env-home)/pycuda/pycuda.bash && pycuda-env $* ; }
google-translate-cli-(){      . $(env-home)/tools/google-translate-cli.bash && google-translate-cli-env $* ; }
pyrtf-(){      . $(env-home)/tools/pyrtf.bash && pyrtf-env $* ; }
mysql_numpy-(){      . $(env-home)/npy/mysql_numpy.bash && mysql_numpy-env $* ; }
pytables-(){      . $(env-home)/npy/pytables.bash && pytables-env $* ; }
virtualbox-(){      . $(env-home)/virtualization/virtualbox.bash && virtualbox-env $* ; }
xen-(){      . $(env-home)/virtualization/xen.bash && xen-env $* ; }
nvgpu-(){      . $(env-home)/virtualization/nvgpu.bash && nvgpu-env $* ; }
rcuda-(){      . $(env-home)/cuda/rcuda.bash && rcuda-env $* ; }
xenserver-(){      . $(env-home)/virtualization/xenserver.bash && xenserver-env $* ; }
h3d-(){      . $(env-home)/graphics/vrml/h3d.bash && h3d-env $* ; }
sqlite3-(){      . $(env-home)/sqlite/sqlite3.bash && sqlite3-env $* ; }
eai-(){      . $(env-home)/graphics/vrml/instant_reality_player/eai.bash && eai-env $* ; }
jcli-(){      . $(env-home)/java/commons/cli/jcli.bash && jcli-env $* ; }
shapedb-(){      . $(env-home)/geant4/geometry/vrml2/shapedb.bash && shapedb-env $* ; }

pycollada-(){      . $(env-home)/graphics/collada/pycollada.bash && pycollada-env $* ; }
collada-(){      . $(env-home)/graphics/collada/collada.bash && collada-env $* ; }
gdmldb-(){      . $(env-home)/geant4/geometry/gdml/gdmldb.bash && gdmldb-env $* ; }
g4beta-(){      . $(env-home)/geant4/g4beta.bash && g4beta-env $* ; }
g4py-(){      . $(env-home)/geant4/g4py/g4py.bash && g4py-env $* ; }
gdml-(){      . $(env-home)/geant4/geometry/gdml/gdml.bash && gdml-env $* ; }
cmake-(){      . $(env-home)/tools/cmake.bash && cmake-env $* ; }
daefile-(){    . $(env-home)/geant4/geometry/daefile.bash && daefile-env $* ; }
dae-(){      . $(env-home)/geant4/geometry/dae.bash && dae-env $* ; }
g4-(){      . $(env-home)/geant4/g4.bash && g4-env $* ; }
panda3d-(){      . $(env-home)/graphics/panda3d/panda3d.bash && panda3d-env $* ; }
meshtool-(){      . $(env-home)/graphics/collada/meshtool.bash && meshtool-env $* ; }
clhep-(){      . $(env-home)/geant4/clhep.bash && clhep-env $* ; }
xercesc-(){      . $(env-home)/xml/xercesc.bash && xercesc-env $* ; }
chromaserver-(){      . $(env-home)/chroma/chromaserver.bash && chromaserver-env $* ; }
webpy-(){      . $(env-home)/webpy/webpy.bash && webpy-env $* ; }
gprof2dot-(){      . $(env-home)/tools/gprof2dot.bash && gprof2dot-env $* ; }
cg-(){      . $(env-home)/graphics/cg/cg.bash && cg-env $* ; }
networkx-(){      . $(env-home)/tools/networkx.bash && networkx-env $* ; }
webglbook-(){      . $(env-home)/graphics/webgl/webglbook.bash && webglbook-env $* ; }
g4daeserver-(){      . $(env-home)/geant4/geometry/g4daeserver/g4daeserver.bash && g4daeserver-env $* ; }
np-(){      . $(env-home)/python/np.bash && np-env $* ; }
threejs-(){      . $(env-home)/graphics/webgl/threejs/threejs.bash && threejs-env $* ; }
hieroglyph-(){      . $(env-home)/doc/hieroglyph.bash && hieroglyph-env $* ; }
s5-(){      . $(env-home)/doc/s5.bash && s5-env $* ; }
sphinxhtmlslide-(){      . $(env-home)/doc/sphinxhtmlslide.bash && sphinxhtmlslide-env $* ; }
rst2s5-(){      . $(env-home)/doc/rst2s5.bash && rst2s5-env $* ; }
rst2pdf-(){      . $(env-home)/doc/rst2pdf.bash && rst2pdf-env $* ; }
reportlab-(){      . $(env-home)/doc/reportlab.bash && reportlab-env $* ; }
vcglib-(){      . $(env-home)/graphics/meshlab/vcglib.bash && vcglib-env $* ; }
qt4-(){      . $(env-home)/ui/qt4.bash && qt4-env $* ; }
slides-(){      . $(env-home)/doc/slides.bash && slides-env $* ; }
eg4dae-(){     . $(env-home)/geant4/geometry/collada/g4dae.bash && eg4dae-env $* ; }
dns-(){      . $(env-home)/sysadmin/dns.bash && dns-env $* ; }
gausstools-(){      . $(env-home)/geant4/geometry/GaussTools/gausstools.bash && gausstools-env $* ; }
vrml-(){      . $(env-home)/geant4/geometry/VRML/vrml.bash && vrml-env $* ; }
mdworker-(){      . $(env-home)/sysadmin/mdworker.bash && mdworker-env $* ; }
lsof-(){      . $(env-home)/sysadmin/lsof.bash && lsof-env $* ; }
meshlabdev-(){      . $(env-home)/graphics/meshlabdev/meshlabdev.bash && meshlabdev-env $* ; }
gitsrc-(){      . $(env-home)/git/gitsrc.bash && gitsrc-env $* ; }
dbus-(){      . $(env-home)/network/dbus.bash && dbus-env $* ; }
geotest-(){      . $(env-home)/geant4/geometry/gdml_example_G02/geotest.bash && geotest-env $* ; }
export-(){      . $(env-home)/geant4/geometry/export/export.bash && export-env $* ; }
osg-(){      . $(env-home)/graphics/openscenegraph/osg.bash && osg-env $* ; }
openvpn-(){      . $(env-home)/network/openvpn.bash && openvpn-env $* ; }
colladadom-(){      . $(env-home)/graphics/collada/colladadom/colladadom.bash && colladadom-env $* ; }
osgdata-(){      . $(env-home)/graphics/openscenegraph/osgdata.bash && osgdata-env $* ; }
colladadomtest-(){      . $(env-home)/graphics/collada/colladadom/testColladaDOM/colladadomtest.bash && colladadomtest-env $* ; }


xquartz-(){      . $(env-home)/gui/xquartz.bash && xquartz-env $* ; }
shrinkwrap-(){      . $(env-home)/python/shrinkwrap/shrinkwrap.bash && shrinkwrap-env $* ; }
pip-(){      . $(env-home)/python/pip.bash && pip-env $* ; }
pygame-(){      . $(env-home)/pygame/pygame.bash && pygame-env $* ; }
dybinst-(){      . $(env-home)/nuwa/dybinst.bash && dybinst-env $* ; }
g4ten-(){      . $(env-home)/geant4/g4ten.bash && g4ten-env $* ; }
zeromq-(){      . $(env-home)/zeromq/zeromq.bash && zeromq-env $* ; }
transformations-(){      . $(env-home)/graphics/transformations/transformations.bash && transformations-env $* ; }
gfxcardstatus-(){      . $(env-home)/cuda/gfxcardstatus.bash && gfxcardstatus-env $* ; }
pyrr-(){      . $(env-home)/graphics/transformations/pyrr.bash && pyrr-env $* ; }
pyopengl-(){      . $(env-home)/graphics/pyopengl/pyopengl.bash && pyopengl-env $* ; }
fcollada-(){      . $(env-home)/graphics/collada/fcollada/fcollada.bash && fcollada-env $* ; }
glumpy-(){      . $(env-home)/graphics/glumpy/glumpy.bash && glumpy-env $* ; }
freeglut-(){      . $(env-home)/graphics/opengl/freeglut/freeglut.bash && freeglut-env $* ; }
g4daeview-(){     . $(env-home)/geant4/geometry/collada/g4daeview/g4daeview.bash && g4daeview-env $* ; }
pytools-(){      . $(env-home)/python/pytools.bash && pytools-env $* ; }
numbers-(){      . $(env-home)/osx/numbers/numbers.bash && numbers-env $* ; }
olxe-(){          . $(env-home)/geant4/examples/lxe/lxe.bash && lxe-env $* ; }

zmqroot-(){      . $(env-home)/zmqroot/zmqroot.bash && zmqroot-env $* ; }
cpl-(){      . $(env-home)/chroma/ChromaPhotonList/cpl.bash && cpl-env $* ; }
otool-(){      . $(env-home)/tools/otool.bash && otool-env $* ; }
echoserver-(){      . $(env-home)/chroma/echoserver/echoserver.bash && echoserver-env $* ; }
specrend-(){      . $(env-home)/graphics/color/specrend/specrend.bash && specrend-env $* ; }
g4view-(){      . $(env-home)/geant4/g4view.bash && g4view-env $* ; }
czrt-(){      . $(env-home)/chroma/ChromaZMQRootTest/czrt.bash && czrt-env $* ; }
csa-(){      . $(env-home)/nuwa/detsim/csa.bash && csa-env $* ; }
nuwacmt-(){      . $(env-home)/nuwa/nuwacmt.bash && nuwacmt-env $* ; }
pyzmq-(){      . $(env-home)/zeromq/pyzmq/pyzmq.bash && pyzmq-env $* ; }
czmq-(){      . $(env-home)/zeromq/czmq/czmq.bash && czmq-env $* ; }
zmq-(){       . $(env-home)/zeromq/zmq/zmq.bash && zmq-env $* ; }

opengl-(){      . $(env-home)/graphics/opengl/opengl.bash && opengl-env $* ; }
atb-(){      . $(env-home)/graphics/atb/atb.bash && atb-env $* ; }
detsim-(){      . $(env-home)/nuwa/detsim/detsim.bash && detsim-env $* ; }
ld-(){      . $(env-home)/tools/ld.bash && ld-env $* ; }
capnproto-(){      . $(env-home)/serialization/capnproto.bash && capnproto-env $* ; }
piwik-(){      . $(env-home)/web/piwik.bash && piwik-env $* ; }
awstats-(){      . $(env-home)/web/awstats.bash && awstats-env $* ; }
vispy-(){      . $(env-home)/graphics/vispy/vispy.bash && vispy-env $* ; }
graphicstools-(){      . $(env-home)/osx/graphicstools/graphicstools.bash && graphicstools-env $* ; }
trac2bitbucket-(){      . $(env-home)/trac/migration/trac2bitbucket.bash && trac2bitbucket-env $* ; }
tracmigrate-(){      . $(env-home)/trac/migration/tracmigrate.bash && tracmigrate-env $* ; }
hgapi-(){      . $(env-home)/hg/hgapi.bash && hgapi-env $* ; }
adm-(){      . $(env-home)/adm/adm.bash && adm-env $* ; }
scmmigrate-(){      . $(env-home)/scm/migration/scmmigrate.bash && scmmigrate-env $* ; }
pysvn-(){      . $(env-home)/svn/pysvn.bash && pysvn-env $* ; }
bitbucket-(){      . $(env-home)/scm/bitbucket/bitbucket.bash && bitbucket-env $* ; }
bitbucketstatic-(){      . $(env-home)/simoncblyth.bitbucket.io/bitbucketstatic.bash && bitbucketstatic-env $* ; }
b2c-(){      . $(env-home)/b2c/b2c.bash && b2c-env $* ; }
rst2html-(){      . $(env-home)/doc/rst2html/rst2html.bash && rst2html-env $* ; }
docutils-(){      . $(env-home)/doc/docutils_/docutils.bash && docutils-env $* ; }
postfix-(){      . $(env-home)/mail/postfix.bash && postfix-env $* ; }
de-(){      . $(env-home)/nuwa/de.bash && de-env $* ; }

npyreader-(){      . $(env-home)/numpy/npyreader.bash && npyreader-env $* ; }
cnpy-(){      . $(env-home)/numpy/cnpy.bash && cnpy-env $* ; }
flup-(){      . $(env-home)/wsgi/flup.bash && flup-env $* ; }
g4daenode-(){      . $(env-home)/geant4/geometry/collada/g4daenode.bash && g4daenode-env $* ; }
presentation-(){      . $(env-home)/presentation/presentation.bash && presentation-env $* ; }
p-(){                 . $(env-home)/presentation/presentation.bash && presentation-env $* ; }
chep-(){              . $(env-home)/presentation/chep/chep.bash && chep-env $* ; }
cepc-(){              . $(env-home)/presentation/cepc/cepc.bash && cepc-env $* ; }
bes3-(){              . $(env-home)/presentation/bes3/bes3.bash && bes3-env $* ; }


dataquality-(){      . $(env-home)/nuwa/dataquality.bash && dataquality-env $* ; }
scenekit-(){      . $(env-home)/graphics/scenekit/scenekit.bash && scenekit-env $* ; }
g4daeplay-(){      . $(env-home)/geant4/geometry/collada/swift/g4daeplay.bash && g4daeplay-env $* ; }
chromacpp-(){      . $(env-home)/chroma/chromacpp/chromacpp.bash && chromacpp-env $* ; }

etherealengine-(){ . $(env-home)/graphics/etherealengine/etherealengine.bash && etherealengine-env $* ; }
webxr-(){          . $(env-home)/graphics/webxr/webxr.bash && webxr-env $* ; }

mocksim-(){      . $(env-home)/geant4/mocksim/mocksim.bash && mocksim-env $* ; }
utilities-(){      . $(env-home)/nuwa/utilities.bash && utilities-env $* ; }
gdc-(){      . $(env-home)/chroma/G4DAEChroma/gdc.bash && gdc-env $* ; }
datamodel-(){      . $(env-home)/nuwa/DataModel/datamodel.bash && datamodel-env $* ; }
datamodeltest-(){      . $(env-home)/nuwa/DataModelTest/datamodeltest.bash && datamodeltest-env $* ; }
rootsys-(){      . $(env-home)/root/rootsys.bash && rootsys-env $* ; }
geant4sys-(){      . $(env-home)/geant4/geant4sys.bash && geant4sys-env $* ; }
mocknuwa-(){      . $(env-home)/nuwa/MockNuWa/mocknuwa.bash && mocknuwa-env $* ; }
gdct-(){      . $(env-home)/chroma/G4DAEChromaTest/gdct.bash && gdct-env $* ; }
cnpytest-(){      . $(env-home)/numpy/cnpytest/cnpytest.bash && cnpytest-env $* ; }
scraper-(){      . $(env-home)/nuwa/scraper/scraper.bash && scraper-env $* ; }
cq-(){      . $(env-home)/nuwa/cq/cq.bash && cq-env $* ; }
libnpy-(){      . $(env-home)/numpy/libnpy.bash && libnpy-env $* ; }
rlibnpy-(){      . $(env-home)/numpy/rlibnpy/rlibnpy.bash && rlibnpy-env $* ; }
cblosc-(){      . $(env-home)/base/compression/blosc/cblosc.bash && cblosc-env $* ; }
mpl-(){      . $(env-home)/matplotlib/mpl.bash && mpl-env $* ; }
jsmn-(){      . $(env-home)/messaging/jsmn.bash && jsmn-env $* ; }
#metal-(){      . $(env-home)/graphics/metal/metal.bash && metal-env $* ; }  ## MOVED TO play-
gason-(){      . $(env-home)/messaging/gason.bash && gason-env $* ; }
testsqlite-(){      . $(env-home)/sqlite/testsqlite/testsqlite.bash && testsqlite-env $* ; }
pythonext-(){      . $(env-home)/python/pythonext/pythonext.bash && pythonext-env $* ; }
rapsqlite-(){      . $(env-home)/sqlite/rapsqlite/rapsqlite.bash && rapsqlite-env $* ; }
cjs-(){      . $(env-home)/messaging/cjson/cjs.bash && cjs-env $* ; }
sqliteswift-(){      . $(env-home)/sqlite/sqliteswift/sqliteswift.bash && sqliteswift-env $* ; }
lineprofiler-(){      . $(env-home)/python/lineprofiler/lineprofiler.bash && lineprofiler-env $* ; }
wt-(){      . $(env-home)/web/wt.bash && wt-env $* ; }
bash-(){      . $(env-home)/base/bash.bash && bash-env $* ; }
vim-(){      . $(env-home)/base/vim/vim.bash && vim-env $* ; }
envcap-(){      . $(env-home)/base/envcap.bash && envcap-env $* ; }
realtime-(){      . $(env-home)/base/time/realtime.bash && realtime-env $* ; }
fdp-(){      . $(env-home)/tools/graphviz/fdp.bash && fdp-env $* ; }
osx_(){      . $(env-home)/osx/osx.bash && osx_env $* ; }
console-(){  . $(env-home)/osx/console.bash && console-env $* ; }
curl-(){     . $(env-home)/tools/curl/curl.bash && curl-env $* ; }
api-(){      . $(env-home)/graphics/api/api.bash && api-env $* ; }
mojo-(){     . $(env-home)/mojo/mojo.bash && mojo-env $* ; }

oppr-(){         . $(env-home)/optix/OppositeRenderer/oppr.bash && oppr-env $* ; }
optixsample1-(){ . $(env-home)/cuda/optix/optix301/sample1manual/optixsample1.bash && optixsample1-env $* ; }


raytrace-(){     . $(env-home)/graphics/raytrace/raytrace.bash && raytrace-env $* ; }
mercurial-(){    . $(env-home)/hg/mercurial.bash && mercurial-env $* ; }
virtualgl-(){    . $(env-home)/graphics/virtualgl/virtualgl.bash && virtualgl-env $* ; }
nvidia-(){       . $(env-home)/graphics/nvidia/nvidia.bash && nvidia-env $* ; }
cudaz-(){        . $(env-home)/cuda/cudaz/cudaz.bash && cudaz-env $* ; }
mesa-(){         . $(env-home)/graphics/opengl/mesa/mesa.bash && mesa-env $* ; }
libpng-(){       . $(env-home)/graphics/libpng/libpng.bash && libpng-env $* ; }
macrosim-(){     . $(env-home)/optix/macrosim/macrosim.bash && macrosim-env $* ; }


vmd-(){          . $(env-home)/optix/vmd/vmd.bash && vmd-env $* ; }
openrl-(){       . $(env-home)/graphics/openrl/openrl.bash && openrl-env $* ; }


cudatex-(){      . $(env-home)/cuda/texture/cudatex.bash && cudatex-env $* ; }
optixtex-(){     . $(env-home)/optix/optixtex/optixtex.bash && optixtex-env $* ; }
unity-(){        . $(env-home)/graphics/unity/unity.bash && unity-env $* ; }
pbs-(){          . $(env-home)/graphics/shading/pbs.bash && pbs-env $* ; }
iray-(){         . $(env-home)/iray/iray.bash && iray-env $* ; }

vl-(){           . $(env-home)/graphics/vl/vl.bash && vl-env $* ; }
vltest-(){       . $(env-home)/graphics/vl/vltest/vltest.bash && vltest-env $* ; }
oglplus-(){      . $(env-home)/graphics/oglplus/oglplus.bash && oglplus-env $* ; }
oglplustest-(){  . $(env-home)/graphics/oglplus/oglplustest/oglplustest.bash && oglplustest-env $* ; }


gl-(){           . $(env-home)/graphics/opengl/gl.bash && gl-env $* ; }
wendy-(){        . $(env-home)/graphics/wendy/wendy.bash && wendy-env $* ; }
basio-(){        . $(env-home)/boost/basio/basio.bash && basio-env $* ; }
asio-(){         . $(env-home)/network/asio/asio.bash && asio-env $* ; }
#numpyserver-(){  . $(env-home)/boost/basio/numpyserver/numpyserver.bash && numpyserver-env $* ; }
numpyserver-(){  . $(env-home)/numpyserver/numpyserver.bash && numpyserver-env $* ; }
photonio-(){     . $(env-home)/graphics/photonio/photonio.bash && photonio-env $* ; }
fishtank-(){     . $(env-home)/graphics/fishtank/fishtank.bash && fishtank-env $* ; }
sdl-(){          . $(env-home)/graphics/sdl/sdl.bash && sdl-env $* ; }
sfml-(){         . $(env-home)/graphics/sfml/sfml.bash && sfml-env $* ; }


bpo-(){          . $(env-home)/boost/bpo/bpo.bash && bpo-env $* ; }
asiozmq-(){      . $(env-home)/network/asiozmq/asiozmq.bash && asiozmq-env $* ; }
asiozmqtest-(){  . $(env-home)/network/asiozmqtest/asiozmqtest.bash && asiozmqtest-env $* ; }
azmq-(){         . $(env-home)/network/azmq/azmq.bash && azmq-env $* ; }
asiosamples-(){  . $(env-home)/boost/basio/asiosamples/asiosamples.bash && asiosamples-env $* ; }
bcfgtest-(){     . $(env-home)/boost/bpo/bcfg/test/bcfgtest.bash && bcfgtest-env $* ; }
hrt-(){          . $(env-home)/graphics/hybrid-rendering-thesis/hrt.bash && hrt-env $* ; }
blogg-(){        . $(env-home)/boost/blogg/blogg.bash && blogg-env $* ; }
ntuwireless-(){  . $(env-home)/admin/ntuwireless.bash && ntuwireless-env $* ; }

bfs-(){          . $(env-home)/boost/bfs/bfs.bash && bfs-env $* ; }
bpt-(){          . $(env-home)/boost/bpt/bpt.bash && bpt-env $* ; }


word-(){         . $(env-home)/tools/word/word.bash && word-env $* ; }
pages-(){        . $(env-home)/tools/pages/pages.bash && pages-env $* ; }
docx-(){         . $(env-home)/tools/docx/docx.bash && docx-env $* ; }
docxbuilder-(){  . $(env-home)/tools/docxbuilder/docxbuilder.bash && docxbuilder-env $* ; }
mono-(){         . $(env-home)/tools/mono/mono.bash && mono-env $* ; }
openxml-(){      . $(env-home)/tools/openxml/openxml.bash && openxml-env $* ; }
argparse-(){     . $(env-home)/python/argparse/argparse.bash && argparse-env $* ; }
hgssh-(){        . $(env-home)/hg/hgssh.bash && hgssh-env $* ; }
imagecapture-(){ . $(env-home)/osx/imagecapture.bash && imagecapture-env $* ; }
preview-(){      . $(env-home)/osx/preview/preview.bash && preview-env $* ; }
brandom-(){      . $(env-home)/boost/random/brandom.bash && brandom-env $* ; }
librocket-(){    . $(env-home)/graphics/gui/librocket/librocket.bash && librocket-env $* ; }

ispm-(){         . $(env-home)/graphics/ispm/ispm.bash && ispm-env $* ; }

thrusthello-(){  . $(env-home)/numerics/thrust/hello/thrusthello.bash && thrusthello-env $* ; }
photonmap-(){    . $(env-home)/graphics/photonmap/photonmap.bash && photonmap-env $* ; }
thrustexamples-(){  . $(env-home)/numerics/thrust/thrustexamples/thrustexamples.bash && thrustexamples-env $* ; }


throgl-(){      . $(env-home)/graphics/thrust_opengl_interop/throgl.bash && throgl-env $* ; }
glfwminimal-(){ . $(env-home)/graphics/glfw/glfwminimal/glfwminimal.bash && glfwminimal-env $* ; }

glewminimal-(){ . $(env-home)/graphics/glew/glewminimal.bash && glewminimal-env $* ; }
glewminimal-(){ . $(env-home)/graphics/glew/glewminimal/glewminimal.bash && glewminimal-env $* ; }
turbovnc-(){    . $(env-home)/network/turbovnc/turbovnc.bash && turbovnc-env $* ; }
gputest-(){     . $(env-home)/network/gputest/gputest.bash && gputest-env $* ; }
java-(){        . $(env-home)/tools/java/java.bash && java-env $* ; }

instancecull-(){      . $(env-home)/graphics/opengl/instancecull.bash && instancecull-env $* ; }
glfwtriangle-(){      . $(env-home)/graphics/glfw/glfwtriangle/glfwtriangle.bash && glfwtriangle-env $* ; }

optixminimal-(){     . $(env-home)/optix/optixminimal/optixminimal.bash && optixminimal-env $* ; }
optixthrust-(){      . $(env-home)/optix/optixthrust/optixthrust.bash && optixthrust-env $* ; }

nvcc-(){      . $(env-home)/cuda/nvcc/nvcc.bash && nvcc-env $* ; }
gloptixthrust-(){      . $(env-home)/optix/gloptixthrust/gloptixthrust.bash && gloptixthrust-env $* ; }
optixthrustnpy-(){      . $(env-home)/optix/optixthrustnpy/optixthrustnpy.bash && optixthrustnpy-env $* ; }
optixthrustuse-(){      . $(env-home)/optix/optixthrustuse/optixthrustuse.bash && optixthrustuse-env $* ; }
detdesc-(){      . $(env-home)/nuwa/detdesc/detdesc.bash && detdesc-env $* ; }
cgal-(){      . $(env-home)/graphics/cgal/cgal.bash && cgal-env $* ; }

pbrt-(){      . $(env-home)/graphics/pbrt/pbrt.bash && pbrt-env $* ; }
ciexyz-(){      . $(env-home)/graphics/ciexyz/ciexyz.bash && ciexyz-env $* ; }
icosahedron-(){      . $(env-home)/graphics/geometry/icosahedron/icosahedron.bash && icosahedron-env $* ; }
refractiveindex-(){      . $(env-home)/physics/refractiveindex/refractiveindex.bash && refractiveindex-env $* ; }
ufunc-(){      . $(env-home)/npy/ufunc/ufunc.bash && ufunc-env $* ; }
lxee-(){      . $(env-home)/geant4/examples/lxee/lxee.bash && lxee-env $* ; }


xcode-(){      . $(env-home)/xcode/xcode.bash && xcode-env $* ; }
g4macports-(){      . $(env-home)/g4/g4macports.bash && g4macports-env $* ; }
g4ex-(){      . $(env-home)/g4/g4ex.bash && g4ex-env $* ; }
lxe-(){      . $(env-home)/optix/lxe/lxe.bash && lxe-env $* ; }

mdls-(){      . $(env-home)/osx/mdls/mdls.bash && mdls-env $* ; }


vxgi-(){      . $(env-home)/graphics/nvidia/vxgi.bash && vxgi-env $* ; }

ios-(){       . $(env-home)/ios/ios.bash && ios-env $* ; }
gpuhep-(){    . $(env-home)/gpuhep/gpuhep.bash && gpuhep-env $* ; }

gtc-(){            . $(env-home)/presentation/gtc2016/gtc.bash && gtc-env $* ; }
vids-(){           . $(env-home)/graphics/ggeoview/vids.bash && vids-env $* ; }
vbox-(){           . $(env-home)/virtualbox/vbox.bash && vbox-env $* ; }

### vr related

vr-(){            . $(env-home)/vr/vr.bash && vr-env $* ; }
ovrminimal-(){    . $(env-home)/vr/ovrminimal/ovrminimal.bash && ovrminimal-env $* ; }
openvr-(){        . $(env-home)/vr/openvr/openvr.bash && openvr-env $* ; }
rift-(){          . $(env-home)/vr/rift/rift.bash && rift-env $* ; }


libgit2-(){       . $(env-home)/git/libgit2.bash && libgit2-env $* ; }
msys2-(){         . $(env-home)/windows/msys2.bash && msys2-env $* ; }
rst2docx-(){      . $(env-home)/doc/docutils_/rst2docx.bash && rst2docx-env $* ; }

# tools

lldb-(){            . $(env-home)/tools/lldb_/lldb.bash && lldb-env $* ; }
cmak-(){            . $(env-home)/tools/cmak.bash && cmak-env $* ; }
openssh-(){         . $(env-home)/tools/openssh/openssh.bash && openssh-env $* ; }
cmakecheck-(){      . $(env-home)/cmakecheck/cmakecheck.bash && cmakecheck-env $* ; }

# windows learning 

win-(){             . $(env-home)/windows/win.bash && win-env $* ; }
gitbash-(){         . $(env-home)/windows/gitbash.bash && gitbash-env $* ; }
importlib-(){       . $(env-home)/windows/importlib.bash && importlib-env $* ; }
importclient-(){    . $(env-home)/windows/importclient/importclient.bash && importclient-env $* ; }
conemu-(){          . $(env-home)/windows/conemu.bash && conemu-env $* ; }
vs-(){              . $(env-home)/windows/vs/vs.bash && vs-env $* ; }
chocolatey-(){      . $(env-home)/windows/chocolatey.bash && chocolatey-env $* ; }
powershell-(){      . $(env-home)/windows/powershell.bash && powershell-env $* ; }
nuget-(){           . $(env-home)/windows/nuget.bash && nuget-env $* ; }
ome-(){             . $(env-home)/windows/ome/ome.bash && ome-env $* ; }
msbuild-(){         . $(env-home)/windows/msbuild/msbuild.bash && msbuild-env $* ; }
g4win-(){           . $(env-home)/g4/g4win.bash && g4win-env $* ; }

# dev projs

vrworks-(){         . $(env-home)/vr/vrworks/vrworks.bash && vrworks-env $* ; }
designworks-(){     . $(env-home)/graphics/nvidia/designworks.bash && designworks-env $* ; }


### opticks dev projs or notes not migrated into opticks repo : as not relevant to users

cudainstall-(){           . $(env-home)/cuda/cudainstall.bash && cudainstall-env $* ; }
bcfg-(){                  . $(env-home)/boostrap/bcfg.bash && bcfg-env $* ; }
ppm-(){                   . $(env-home)/graphics/ppm/ppm.bash && ppm-env $* ; }
ppmfast-(){               . $(env-home)/graphics/ppmfast/ppmfast.bash && ppmfast-env $* ; }
gleqtest-(){              . $(env-home)/graphics/glfw/gleqtest/gleqtest.bash && gleqtest-env $* ; }

csg-(){                   . $(env-home)/graphics/csg/csg.bash && csg-env $* ; }
pmt-(){                   . $(env-home)/nuwa/detdesc/pmt/pmt.bash && pmt-env $* ; }
okt-(){                   . $(env-home)/optickstute/okt.bash && okt-env $* ; }
brc-(){                   . $(env-home)/boostrapclient/brc.bash && brc-env $* ; }
npc-(){                   . $(env-home)/numerics/npyclient/npc.bash && npc-env $* ; }
omc-(){                   . $(env-home)/graphics/openmeshclient/omc.bash && omc-env $* ; }
ggeodev-(){               . $(env-home)/optix/ggeo/ggeodev.bash && ggeodev-env $* ; }
proj-(){                  . $(env-home)/base/proj.bash && proj-env $* ; }
solarmd5-(){              . $(env-home)/tools/solarmd5/solarmd5.bash && solarmd5-env $* ; }
optickswin-(){            . $(env-home)/optickswin.bash && optickswin-env $* ; }
opticks-failed-build-(){  . $(env-home)/opticks-failed-build.bash ; }
opticksdev-(){            . $(env-home)/opticksdev.bash ; }
g4opgen-(){               . $(env-home)/geant4/g4op/g4opgen.bash && g4opgen-env $* ; }
cmakex-(){                . $(env-home)/tools/cmakex.bash && cmakex-env $* ; }

g4d-(){                   . $HOME/g4dae/g4d.bash && g4d-env $* ; } # OTHER repo interloper

optixsamples-(){ . $(env-home)/optix/optixsamples.bash && optixsamples-env $* ; }
glfwtest-(){     . $(env-home)/graphics/glfw/glfwtest/glfwtest.bash && glfwtest-env $* ; }
optixtest-(){    . $(env-home)/optix/OptiXTest/optixtest.bash && optixtest-env $* ; }
assimptest-(){   . $(env-home)/graphics/assimp/AssimpTest/assimptest.bash && assimptest-env $* ; }

imguitest-(){    . $(env-home)/graphics/gui/imguitest/imguitest.bash && imguitest-env $* ; }
g4op-(){         . $(env-home)/geant4/g4op/g4op.bash && g4op-env $* ; }







#### opticks externals ###  TODO: **copy** into externals with opticks-xcollect, once spawned can remove non-essentials (dev notes etc..)  
#
#boost-(){           . $(env-home)/boost/boost.bash && boost-env $* ; }
#glm-(){             . $(env-home)/graphics/glm/glm.bash && glm-env $* ; }
#plog-(){            . $(env-home)/tools/plog/plog.bash && plog-env $* ; }
#
#gleq-(){            . $(env-home)/graphics/gleq/gleq.bash && gleq-env $* ; }
#glfw-(){            . $(env-home)/graphics/glfw/glfw.bash && glfw-env $* ; }
#glew-(){            . $(env-home)/graphics/glew/glew.bash && glew-env $* ; }
#imgui-(){           . $(env-home)/graphics/gui/imgui/imgui.bash && imgui-env $* ; }
#
#assimp-(){          . $(env-home)/graphics/assimp/assimp.bash && assimp-env $* ; }
#openmesh-(){        . $(env-home)/graphics/openmesh/openmesh.bash && openmesh-env $* ; }
#
#cuda-(){            . $(env-home)/cuda/cuda.bash && cuda-env $* ; }
#thrust-(){          . $(env-home)/numerics/thrust/thrust.bash && thrust-env $* ; }
#optix-(){           . $(env-home)/optix/optix.bash && optix-env $* ; }
#
#xercesc-(){         . $(env-home)/xml/xercesc/xercesc.bash && xercesc-env $* ; }
#g4-(){              . $(env-home)/g4/g4.bash && g4-env $* ; }
#
#### opticks infrastructure/launchers ###  other than opticks.bash **move** into externals (?) 
#
#opticks-(){         . $(env-home)/opticks.bash && opticks-env $* ; }
#opticksdata-(){     . $(env-home)/opticksdata.bash && opticksdata-env $* ; }
#ggv-(){             . $(env-home)/ggeoview/ggv.bash && ggv-env $* ; }
#op-(){              . $(env-home)/bin/op.sh ; }
#
#### opticks projs ###  **moved** all projs into top level folders
#
#sysrap-(){          . $(env-home)/sysrap/sysrap.bash && sysrap-env $* ; }
#brap-(){            . $(env-home)/boostrap/brap.bash && brap-env $* ; }
#npy-(){             . $(env-home)/opticksnpy/npy.bash && npy-env $* ; }
#okc-(){             . $(env-home)/optickscore/okc.bash && okc-env $* ; }
#
#ggeo-(){            . $(env-home)/ggeo/ggeo.bash && ggeo-env $* ; }
#assimprap-(){       . $(env-home)/assimprap/assimprap.bash && assimprap-env $* ; }
#openmeshrap-(){     . $(env-home)/openmeshrap/openmeshrap.bash && openmeshrap-env $* ; }
#opticksgeo-(){      . $(env-home)/opticksgeo/opticksgeo.bash && opticksgeo-env $* ; }
#
#oglrap-(){          . $(env-home)/oglrap/oglrap.bash && oglrap-env $* ; }
#cudarap-(){         . $(env-home)/cudarap/cudarap.bash && cudarap-env $* ; }
#thrustrap-(){       . $(env-home)/thrustrap/thrustrap.bash && thrustrap-env $* ; }
#optixrap-(){        . $(env-home)/optixrap/optixrap.bash && optixrap-env $* ; }
#
#opticksop-(){       . $(env-home)/opticksop/opticksop.bash && opticksop-env $* ; }
#opticksgl-(){       . $(env-home)/opticksgl/opticksgl.bash && opticksgl-env $* ; }
#ggeoview-(){        . $(env-home)/ggeoview/ggeoview.bash && ggeoview-env $* ; }
#cfg4-(){            . $(env-home)/cfg4/cfg4.bash && cfg4-env $* ; }
#
#
#sniper-(){      . $(env-home)/juno/sniper/sniper.bash && sniper-env $* ; }
#offline-(){      . $(env-home)/juno/offline/offline.bash && offline-env $* ; }
#juno-(){      . $(env-home)/juno/juno.bash && juno-env $* ; }
gltf-(){      . $(env-home)/graphics/gltf/gltf.bash && gltf-env $* ; }
filament-(){      . $(env-home)/graphics/filament/filament.bash && filament-env $* ; }
unreal-(){      . $(env-home)/graphics/unreal/unreal.bash && unreal-env $* ; }
steamvr-(){      . $(env-home)/vr/steamvr/steamvr.bash && steamvr-env $* ; }
engine-(){      . $(env-home)/graphics/engine/engine.bash && engine-env $* ; }
ssd-(){      . $(env-home)/hardware/ssd/ssd.bash && ssd-env $* ; }
realpath-(){      . $(env-home)/tools/realpath/realpath.bash && realpath-env $* ; }
vulkan-(){      . $(env-home)/graphics/vulkan/vulkan.bash && vulkan-env $* ; }
ioproc-(){      . $(env-home)/doc/ioproc/ioproc.bash && ioproc-env $* ; }
tcpdump-(){      . $(env-home)/tools/tcpdump.bash && tcpdump-env $* ; }
rootnumpy-(){      . $(env-home)/root/rootnumpy/rootnumpy.bash && rootnumpy-env $* ; }
vecgeom-(){      . $(env-home)/geometry/vecgeom/vecgeom.bash && vecgeom-env $* ; }
sso-(){      . $(env-home)/network/sso/sso.bash && sso-env $* ; }
embree-(){      . $(env-home)/embree/embree.bash && embree-env $* ; }
rst-(){      . $(env-home)/tools/rst/rst.bash && rst-env $* ; }
intersect-(){      . $(env-home)/graphics/intersect/intersect.bash && intersect-env $* ; }
csgtools-(){      . $(env-home)/graphics/csg/csgtools/csgtools.bash && csgtools-env $* ; }
csgformat-(){      . $(env-home)/graphics/csg/csgformat/csgformat.bash && csgformat-env $* ; }
influxdb-(){      . $(env-home)/db/influxdb/influxdb.bash && influxdb-env $* ; }
ccsg-(){      . $(env-home)/env/graphics/csg/ccsg.bash && ccsg-env $* ; }
firerays-(){      . $(env-home)/firerays/firerays.bash && firerays-env $* ; }
csgjscpp-(){      . $(env-home)/graphics/csg/csgjscpp/csgjscpp.bash && csgjscpp-env $* ; }
isoex-(){      . $(env-home)/graphics/csg/isoex.bash && isoex-env $* ; }
piston-(){      . $(env-home)/graphics/piston/piston.bash && piston-env $* ; }
pymcubes-(){      . $(env-home)/graphics/csg/pymcubes.bash && pymcubes-env $* ; }
openvdb-(){      . $(env-home)/graphics/openvdb/openvdb.bash && openvdb-env $* ; }
isosurface-(){      . $(env-home)/graphics/isosurface/isosurface.bash && isosurface-env $* ; }
dualcontouring-(){      . $(env-home)/graphics/isosurface/dualcontouring.bash && dualcontouring-env $* ; }
dcs-(){      . $(env-home)/graphics/isosurface/dualcontouringsample/dcs.bash && dcs-env $* ; }
octree-(){      . $(env-home)/graphics/octree/octree.bash && octree-env $* ; }
mortonlib-(){      . $(env-home)/graphics/mortonlib/mortonlib.bash && mortonlib-env $* ; }
isooctree-(){      . $(env-home)/graphics/isosurface/isooctree.bash && isooctree-env $* ; }
poissonrecon-(){      . $(env-home)/graphics/isosurface/poissonrecon.bash && poissonrecon-env $* ; }
pcl-(){      . $(env-home)/graphics/isosurface/pcl.bash && pcl-env $* ; }
dctatwood-(){      . $(env-home)/graphics/isosurface/dctatwood.bash && dctatwood-env $* ; }
implicitmesher-(){      . $(env-home)/graphics/isosurface/implicitmesher.bash && implicitmesher-env $* ; }
opensource-(){      . $(env-home)/strategy/opensource.bash && opensource-env $* ; }
sdf-(){      . $(env-home)/graphics/isosurface/sdf.bash && sdf-env $* ; }
povray-(){      . $(env-home)/graphics/povray/povray.bash && povray-env $* ; }
scene-(){      . $(env-home)/graphics/scene/scene.bash && scene-env $* ; }
octane-(){      . $(env-home)/graphics/octane/octane.bash && octane-env $* ; }
powervr-(){      . $(env-home)/graphics/powervr/powervr.bash && powervr-env $* ; }
csgparametric-(){      . $(env-home)/graphics/csg/csgparametric.bash && csgparametric-env $* ; }
openscad-(){      . $(env-home)/graphics/csg/openscad.bash && openscad-env $* ; }
gate-(){      . $(env-home)/geant4/gate/gate.bash && gate-env $* ; }
xrt-(){      . $(env-home)/graphics/xrt/xrt.bash && xrt-env $* ; }
renderer-(){      . $(env-home)/graphics/renderer/renderer.bash && renderer-env $* ; }
renderman-(){      . $(env-home)/graphics/renderman/renderman.bash && renderman-env $* ; }
yoctogl-(){      . $(env-home)/graphics/yoctogl/yoctogl.bash && yoctogl-env $* ; }
sympy-(){      . $(env-home)/npy/sympy/sympy.bash && sympy-env $* ; }
gvdb-(){      . $(env-home)/graphics/gvdb/gvdb.bash && gvdb-env $* ; }
svn2git-(){      . $(env-home)/adm/svn2git.bash && svn2git-env $* ; }
csgbsp-(){      . $(env-home)/graphics/csg/csgbsp/csgbsp.bash && csgbsp-env $* ; }
csgbbox-(){      . $(env-home)/graphics/csg/csgbbox.bash && csgbbox-env $* ; }
carve-(){      . $(env-home)/graphics/csg/carve.bash && carve-env $* ; }
libcaca-(){      . $(env-home)/graphics/txt/libcaca.bash && libcaca-env $* ; }
gts-(){      . $(env-home)/graphics/gts/gts.bash && gts-env $* ; }
cosinekitty-(){      . $(env-home)/graphics/cosinekitty.bash && cosinekitty-env $* ; }
quartic-(){      . $(env-home)/geometry/quartic/quartic.bash && quartic-env $* ; }
gems-(){      . $(env-home)/graphics/gems/gems.bash && gems-env $* ; }
rayce-(){      . $(env-home)/graphics/rayce/rayce.bash && rayce-env $* ; }
mountains-(){      . $(env-home)/graphics/opengl/mountains/mountains.bash && mountains-env $* ; }
txf-(){      . $(env-home)/graphics/opengl/txf/txf.bash && txf-env $* ; }
instance-(){      . $(env-home)/graphics/opengl/instance/instance.bash && instance-env $* ; }
instcull-(){      . $(env-home)/graphics/opengl/instcull/instcull.bash && instcull-env $* ; }
oas-(){      . $(env-home)/graphics/optix_advanced_samples/oas.bash && oas-env $* ; }
nature-(){      . $(env-home)/graphics/opengl/nature/nature.bash && nature-env $* ; }
icu-(){      . $(env-home)/intro_to_cuda/icu.bash && icu-env $* ; }
ffmpeg-(){      . $(env-home)/video/ffmpeg/ffmpeg.bash && ffmpeg-env $* ; }
nvenc-(){      . $(env-home)/graphics/nvidia/nvenc.bash && nvenc-env $* ; }
capturesdk-(){      . $(env-home)/graphics/nvidia/capturesdk.bash && capturesdk-env $* ; }
yum-(){      . $(env-home)/tools/yum/yum.bash && yum-env $* ; }
nasm-(){      . $(env-home)/tools/nasm.bash && nasm-env $* ; }
x264-(){      . $(env-home)/video/x264.bash && x264-env $* ; }
egl-(){      . $(env-home)/graphics/egl/egl.bash && egl-env $* ; }
macos-(){      . $(env-home)/osx/macos.bash && macos-env $* ; }

dotfiler-(){      . $(env-home)/tools/dotfiler.bash && dotfiler-env $* ; }




sphinxtest-(){      . $(env-home)/doc/sphinxtest.bash && sphinxtest-env $* ; }
gitsplit-(){      . $(env-home)/adm/gitsplit.bash && gitsplit-env $* ; }
gitfilter-(){      . $(env-home)/adm/gitfilter.bash && gitfilter-env $* ; }
terminal-(){      . $(env-home)/base/terminal/terminal.bash && terminal-env $* ; }
dxr-(){      . $(env-home)/graphics/directx/dxr.bash && dxr-env $* ; }
fastexport-(){      . $(env-home)/tools/hg2git/fastexport.bash && fastexport-env $* ; }
edefaults-(){      . $(env-home)/osx/defaults/edefaults.bash && edefaults-env $* ; }
clt-(){      . $(env-home)/xcode/commandlinetools/clt.bash && clt-env $* ; }
conda-(){      . $(env-home)/tools/conda.bash && conda-env $* ; }
spritekit-(){      . $(env-home)/graphics/spritekit/spritekit.bash && spritekit-env $* ; }
ditto-(){      . $(env-home)/osx/ditto.bash && ditto-env $* ; }

rsync-(){      . $(env-home)/tools/rsync.bash && rsync-env $* ; }


docker-(){      . $(env-home)/tools/docker.bash && docker-env $* ; }
cocoapods-(){      . $(env-home)/tools/cocoapods.bash && cocoapods-env $* ; }
carthage-(){      . $(env-home)/tools/carthage.bash && carthage-env $* ; }
brew-(){      . $(env-home)/tools/brew.bash && brew-env $* ; }
open-(){      . $(env-home)/osx/open.bash && open-env $* ; }
bcm-(){      . $(env-home)/tools/bcm.bash && bcm-env $* ; }
cct-(){      . $(env-home)/cuda/cmake_cuda_tests/cct.bash && cct-env $* ; }
cuda_samples-(){      . $(env-home)/cuda/cuda_samples.bash && cuda_samples-env $* ; }
cudasamples-(){      . $(env-home)/cuda/cudasamples.bash && cudasamples-env $* ; }
gltfkit-(){      . $(env-home)/graphics/gltf/gltfkit.bash && gltfkit-env $* ; }
gltfscenekit-(){      . $(env-home)/graphics/gltf/gltfscenekit.bash && gltfscenekit-env $* ; }
slack-(){      . $(env-home)/comms/slack/slack.bash && slack-env $* ; }
gio-(){      . $(env-home)/comms/groups_io_opticks/gio.bash && gio-env $* ; }
gitforwindows-(){      . $(env-home)/windows/gitforwindows.bash && gitforwindows-env $* ; }
optickswin2-(){      . $(env-home)/windows/optickswin/optickswin2.bash && optickswin2-env $* ; }
obs-(){      . $(env-home)/video/obs-studio/obs.bash && obs-env $* ; }
gnome-(){      . $(env-home)/linux/gnome/gnome.bash && gnome-env $* ; }
vlc-(){      . $(env-home)/video/vlc/vlc.bash && vlc-env $* ; }
imovie-(){      . $(env-home)/video/imovie/imovie.bash && imovie-env $* ; }
groupadd-(){      . $(env-home)/linux/groupadd.bash && groupadd-env $* ; }
vgp-(){      . $(env-home)/graphics/Vulkan-glTF-PBR/vgp.bash && vgp-env $* ; }
swv-(){      . $(env-home)/graphics/SaschaWillemsVulkan/swv.bash && swv-env $* ; }
goofit-(){      . $(env-home)/goofit/goofit.bash && goofit-env $* ; }
numba-(){      . $(env-home)/numerics/numba/numba.bash && numba-env $* ; }
sxmc-(){      . $(env-home)/fit/sxmc.bash && sxmc-env $* ; }
usdz-(){      . $(env-home)/graphics/usdz/usdz.bash && usdz-env $* ; }
theano-(){      . $(env-home)/numerics/theano/theano.bash && theano-env $* ; }
gpufit-(){      . $(env-home)/fit/gpufit.bash && gpufit-env $* ; }
hydra-(){      . $(env-home)/fit/hydra.bash && hydra-env $* ; }
pifi-(){      . $(env-home)/opticks/pifi.bash && pifi-env $* ; }
mcmc-(){      . $(env-home)/numerics/mcmc/mcmc.bash && mcmc-env $* ; }
rng-(){      . $(env-home)/numerics/rng/rng.bash && rng-env $* ; }
rust-(){      . $(env-home)/rust/rust.bash && rust-env $* ; }
bwasty-(){      . $(env-home)/graphics/gltf/viewer/bwasty.bash && bwasty-env $* ; }
carpentry-(){      . $(env-home)/software/carpentry.bash && carpentry-env $* ; }
cusolver-(){      . $(env-home)/cuda/cusolver/cusolver.bash && cusolver-env $* ; }
recon-(){      . $(env-home)/recon/recon.bash && recon-env $* ; }
minuit2-(){      . $(env-home)/recon/minuit2.bash && minuit2-env $* ; }
rpath-(){      . $(env-home)/tools/cmake/rpath.bash && rpath-env $* ; }
pandas-(){      . $(env-home)/numerics/pandas/pandas.bash && pandas-env $* ; }
epjconf-(){      . $(env-home)/doc/epjconf/epjconf.bash && epjconf-env $* ; }
sed-(){      . $(env-home)/tools/sed.bash && sed-env $* ; }
scp-(){      . $(env-home)/tools/scp.bash && scp-env $* ; }
ctest-(){      . $(env-home)/tools/ctest.bash && ctest-env $* ; }
firefox-(){      . $(env-home)/tools/firefox.bash && firefox-env $* ; }

pipeline-(){      . $(env-home)/graphics/pipeline/pipeline.bash && pipeline-env $* ; }
applegpu-(){      . $(env-home)/graphics/apple/applegpu.bash && applegpu-env $* ; }

license-(){      . $(env-home)/license/license.bash && license-env $* ; }
synergy-(){      . $(env-home)/tools/synergy.bash && synergy-env $* ; }
dnf-(){      . $(env-home)/tools/dnf/dnf.bash && dnf-env $* ; }
centos-(){      . $(env-home)/tools/centos.bash && centos-env $* ; }
grub-(){      . $(env-home)/boot/grub.bash && grub-env $* ; }
rtow-(){      . $(env-home)/graphics/RTOW-OptiX/rtow.bash && rtow-env $* ; }
cli11-(){      . $(env-home)/tools/cli/cli11.bash && cli11-env $* ; }
argh-(){      . $(env-home)/tools/cli/argh.bash && argh-env $* ; }
devil-(){      . $(env-home)/graphics/image/devil.bash && devil-env $* ; }
gameworks-(){      . $(env-home)/graphics/nvidia/gameworks.bash && gameworks-env $* ; }

nvml-(){      . $(env-home)/graphics/nvidia/nvml.bash && nvml-env $* ; }
visrtx-(){      . $(env-home)/graphics/nvidia/visrtx.bash && visrtx-env $* ; }
equirect-(){      . $(env-home)/graphics/opengl/equirect.bash && equirect-env $* ; }
ml-(){      . $(env-home)/ai/ml.bash && ml-env $* ; }
appleml-(){ . $(env-home)/ai/appleml.bash && appleml-env $* ; }
vr-(){      . $(env-home)/graphics/vr/vr.bash && vr-env $* ; }
tree-(){      . $(env-home)/adt/tree.bash && tree-env $* ; }
gdb-(){      . $(env-home)/tools/gdb.bash && gdb-env $* ; }

mermaid-(){      . $(env-home)/tools/mermaid/mermaid.bash && mermaid-env $* ; }
tf-(){      . $(env-home)/tools/tensorflow/tf.bash && tf-env $* ; }
keras-(){      . $(env-home)/ai/keras.bash && keras-env $* ; }
igprof-(){      . $(env-home)/tools/igprof.bash && igprof-env $* ; }
mapd-(){      . $(env-home)/tools/mapd.bash && mapd-env $* ; }
licensehd-(){      . $(env-home)/tools/licensehd.bash && licensehd-env $* ; } 
hg2git-(){      . $(env-home)/tools/hg2git/hg2git.bash && hg2git-env $* ; }
mdl-(){      . $(env-home)/graphics/nvidia/mdl.bash && mdl-env $* ; }
nest-(){      . $(env-home)/scintillation/nest/nest.bash && nest-env $* ; }
lighthouse2-(){      . $(env-home)/graphics/lighthouse2/lighthouse2.bash && lighthouse2-env $* ; }
optix7sandbox-(){      . $(env-home)/graphics/optix7/optix7sandbox.bash && optix7sandbox-env $* ; }

pkg-config-(){      . $(env-home)/tools/pkg-config.bash && pkg-config-env $* ; }
hgexporttool-(){      . $(env-home)/tools/hg2git/hgexporttool.bash && hgexporttool-env $* ; }
gitsvn-(){      . $(env-home)/tools/gitsvn.bash && gitsvn-env $* ; }
github-(){      . $(env-home)/scm/github/github.bash && github-env $* ; }
llgl-(){      . $(env-home)/graphics/llgl/llgl.bash && llgl-env $* ; }
vgpu-(){      . $(env-home)/graphics/vgpu/vgpu.bash && vgpu-env $* ; }
bgfx-(){      . $(env-home)/graphics/bgfx/bgfx.bash && bgfx-env $* ; }
dileng-(){      . $(env-home)/graphics/dileng/dileng.bash && dileng-env $* ; }
mayavi-(){      . $(env-home)/graphics/mayavi/mayavi.bash && mayavi-env $* ; }
condaforge-(){      . $(env-home)/python/condaforge.bash && condaforge-env $* ; }
pyvista-(){      . $(env-home)/graphics/pyvista_/pyvista.bash && pyvista-env $* ; }
radeonrays-(){      . $(env-home)/graphics/amd/radeonrays.bash && radeonrays-env $* ; }
vulkanrt-(){      . $(env-home)/graphics/vulkan/vulkanrt.bash && vulkanrt-env $* ; }
intelrt-(){      . $(env-home)/graphics/intel/intelrt.bash && intelrt-env $* ; }
ospray-(){      . $(env-home)/embree/ospray.bash && ospray-env $* ; }
plotoptix-(){      . $(env-home)/graphics/optix7/plotoptix.bash && plotoptix-env $* ; }
slurm-(){      . $(env-home)/batch/slurm.bash && slurm-env $* ; }
osl-(){      . $(env-home)/graphics/osl/osl.bash && osl-env $* ; }
pimpl-(){      . $(env-home)/design/pimpl.bash && pimpl-env $* ; }
oap-(){      . $(env-home)/graphics/optix_apps/oap.bash && oap-env $* ; }
py3-(){      . $(env-home)/python/py3/py3.bash && py3-env $* ; }
nng-(){      . $(env-home)/network/nng.bash && nng-env $* ; }
asyncio-(){      . $(env-home)/python/asyncio/asyncio.bash && asyncio-env $* ; }
threading-(){      . $(env-home)/cpprun/threading/threading.bash && threading-env $* ; }
gpp-(){      . $(env-home)/garfieldpp/gpp.bash && gpp-env $* ; }
cel-(){      . $(env-home)/gpuhep/celeritas/cel.bash && cel-env $* ; }
corsika-(){      . $(env-home)/corsika/corsika.bash && corsika-env $* ; }
stb-(){      . $(env-home)/graphics/stb/stb.bash && stb-env $* ; }
useradd-(){      . $(env-home)/tools/useradd.bash && useradd-env $* ; }
chrt-(){      . $(env-home)/graphics/chameleonrt/chrt.bash && chrt-env $* ; }
rt-(){      . $(env-home)/graphics/rt/rt.bash && rt-env $* ; }
g4ck-(){      . $(env-home)/geant4/g4op/g4ck.bash && g4ck-env $* ; }
aidainnova-(){      . $(env-home)/proj/aidainnova.bash && aidainnova-env $* ; }
k4-(){         . $(env-home)/proj/key4hep/k4.bash && k4-env $* ; }
e4-(){      . $(env-home)/proj/key4hep/e4.bash && e4-env $* ; }
podio-(){      . $(env-home)/proj/key4hep/podio.bash && podio-env $* ; }
sio-(){      . $(env-home)/proj/key4hep/sio.bash && sio-env $* ; }
acts-(){      . $(env-home)/proj/key4hep/acts.bash && acts-env $* ; }
zike-(){      . $(env-home)/proj/zike.bash && zike-env $* ; }
liyu-(){      . $(env-home)/proj/liyu.bash && liyu-env $* ; }
rs-(){        . $(env-home)/proj/Rich_Simplified/rs.bash && rs-env $* ; }
ai-(){        . $(env-home)/ai/ai.bash && ai-env $* ; }
pytorch-(){      . $(env-home)/pytorch/pytorch.bash && pytorch-env $* ; }
usd-(){       . $(env-home)/graphics/usd/usd.bash && usd-env $* ; }

cats-(){      . $(env-home)/cats/cats.bash && cats-env $* ; }
slurm-(){     . $(env-home)/slurm/slurm.bash && slurm-env $* ; }
leak-(){     . $(env-home)/tools/leak.bash && leak-env $* ; }
gitlab-(){     . $(env-home)/tools/gitlab.bash && gitlab-env $* ; }
base64-(){     . $(env-home)/tools/base64.bash && base64-env $* ; }



