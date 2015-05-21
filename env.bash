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


env-touch-index(){
   # recursivly touch existing index.rst going up the directory tree from PWD
   # for pursuading sphinx to rebuild a deep page
   local dir=${1:-$PWD}
   [ "$dir" == "/" ] && return 
   local idx=$dir/index.rst
   [ -f "$idx" ] && echo $idx && touch $idx 
   $FUNCNAME $(dirname $dir)
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


u_(){ 
   # mapping directory to url 
   local dir=${1:-$PWD}
   local abs=$(realpath $dir)   # see ~/e/tools/realpath/
   case $abs in
           ${ENV_HOME}) echo http://localhost/env_notes/ ;; 
          ${ENV_HOME}*) echo http://localhost/env_notes/${abs/$ENV_HOME\/}/ ;; 
      ${WORKFLOW_HOME}) echo http://localhost/w/ ;; 
     ${WORKFLOW_HOME}*) echo http://localhost/w/${abs/$WORKFLOW_HOME\/}/ ;; 
        ${HEPREZ_HOME}) echo http://dayabay.phys.ntu.edu.tw/h/ ;; 
       ${HEPREZ_HOME}*) echo http://dayabay.phys.ntu.edu.tw/h/${abs/$HEPREZ_HOME\/}/ ;; 
      ${DYBGAUDI_HOME}) echo http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/ ;; 
     ${DYBGAUDI_HOME}*) echo http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/${abs/$DYBGAUDI_HOME\/}/ ;; 
                     *) echo http://www.google.com ;;
   esac
}
u(){ 
   local url=$(u_ $*)
   echo $msg open URL corresponding to PWD $PWD : $url
   [ "$(uname)" == "Darwin" ] && open $url 
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
pd(){ env-para ; pwd ; }

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

env-gen(){ func-;func-gen $*;}

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
  svn-
 
  PATH=$(env-home)/bin:$PATH
 
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
svnprecommit-(){      . $(env-home)/svn/svnprecommit.bash && svnprecommit-env $* ; }
dcs-(){      . $(env-home)/offline/dcs.bash && dcs-env $* ; }
#scr-(){      . $(env-home)/offline/scr.bash && scr-env $* ; }
drush-(){      . $(env-home)/drush/drush.bash && drush-env $* ; }
videola-(){  . $(env-home)/videola/videola.bash && videola-env $* ; }
mediamosa-(){      . $(env-home)/mediamosa/mediamosa.bash && mediamosa-env $* ; }
tcpdump-(){      . $(env-home)/base/tcpdump.bash && tcpdump-env $* ; }
svnlog-(){      . $(env-home)/tools/svnlog.bash && svnlog-env $* ; }
macports-(){      . $(env-home)/base/macports.bash && macports-env $* ; }
xsd-(){      . $(env-home)/xml/xsd.bash && xsd-env $* ; }
bdbxml-(){      . $(env-home)/db/bdbxml.bash && bdbxml-env $* ; }
swig-(){      . $(env-home)/swig/swig.bash && swig-env $* ; }
swigbuild-(){      . $(env-home)/swig/swigbuild.bash && swigbuild-env $* ; }
viruscheck-(){      . $(env-home)/admin/viruscheck.bash && viruscheck-env $* ; }
dbxml-(){      . $(env-home)/db/dbxml.bash && dbxml-env $* ; }
boost-(){      . $(env-home)/boost/boost.bash && boost-env $* ; }
boost-(){      . $(env-home)/boost/boost.bash && boost-env $* ; }
boostlog-(){      . $(env-home)/boost/boostlog.bash && boostlog-env $* ; }
docutils-(){      . $(env-home)/python/docutils.bash && docutils-env $* ; }
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
vgpu-(){      . $(env-home)/virtualization/vgpu.bash && vgpu-env $* ; }
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
g4dae-(){      . $(env-home)/geant4/geometry/collada/g4dae.bash && g4dae-env $* ; }
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
cuda-(){      . $(env-home)/cuda/cuda.bash && cuda-env $* ; }
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
lxe-(){          . $(env-home)/geant4/examples/lxe/lxe.bash && lxe-env $* ; }

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
bitbucketstatic-(){      . $(env-home)/simoncblyth.bitbucket.org/bitbucketstatic.bash && bitbucketstatic-env $* ; }
b2c-(){      . $(env-home)/b2c/b2c.bash && b2c-env $* ; }
rst2html-(){      . $(env-home)/doc/rst2html/rst2html.bash && rst2html-env $* ; }
docutils-(){      . $(env-home)/doc/docutils/docutils.bash && docutils-env $* ; }
postfix-(){      . $(env-home)/mail/postfix.bash && postfix-env $* ; }
de-(){      . $(env-home)/nuwa/de.bash && de-env $* ; }

npyreader-(){      . $(env-home)/numpy/npyreader.bash && npyreader-env $* ; }
cnpy-(){      . $(env-home)/numpy/cnpy.bash && cnpy-env $* ; }
flup-(){      . $(env-home)/wsgi/flup.bash && flup-env $* ; }
g4daenode-(){      . $(env-home)/geant4/geometry/collada/g4daenode.bash && g4daenode-env $* ; }
presentation-(){      . $(env-home)/presentation/presentation.bash && presentation-env $* ; }
dataquality-(){      . $(env-home)/nuwa/dataquality.bash && dataquality-env $* ; }
scenekit-(){      . $(env-home)/graphics/scenekit/scenekit.bash && scenekit-env $* ; }
g4daeplay-(){      . $(env-home)/geant4/geometry/collada/swift/g4daeplay.bash && g4daeplay-env $* ; }
chromacpp-(){      . $(env-home)/chroma/chromacpp/chromacpp.bash && chromacpp-env $* ; }
xercesc-(){      . $(env-home)/xml/xercesc/xercesc.bash && xercesc-env $* ; }
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
metal-(){      . $(env-home)/graphics/metal/metal.bash && metal-env $* ; }
gason-(){      . $(env-home)/messaging/gason.bash && gason-env $* ; }
testsqlite-(){      . $(env-home)/sqlite/testsqlite/testsqlite.bash && testsqlite-env $* ; }
pythonext-(){      . $(env-home)/python/pythonext/pythonext.bash && pythonext-env $* ; }
rapsqlite-(){      . $(env-home)/sqlite/rapsqlite/rapsqlite.bash && rapsqlite-env $* ; }
cjs-(){      . $(env-home)/messaging/cjson/cjs.bash && cjs-env $* ; }
sqliteswift-(){      . $(env-home)/sqlite/sqliteswift/sqliteswift.bash && sqliteswift-env $* ; }
lineprofiler-(){      . $(env-home)/python/lineprofiler/lineprofiler.bash && lineprofiler-env $* ; }
wt-(){      . $(env-home)/web/wt.bash && wt-env $* ; }
vim-(){      . $(env-home)/base/vim/vim.bash && vim-env $* ; }
envcap-(){      . $(env-home)/base/envcap.bash && envcap-env $* ; }
realtime-(){      . $(env-home)/base/time/realtime.bash && realtime-env $* ; }
fdp-(){      . $(env-home)/tools/graphviz/fdp.bash && fdp-env $* ; }
osx-(){      . $(env-home)/osx/osx.bash && osx-env $* ; }
optix-(){      . $(env-home)/optix/optix.bash && optix-env $* ; }
oppr-(){      . $(env-home)/optix/OppositeRenderer/oppr.bash && oppr-env $* ; }
optixsample1-(){      . $(env-home)/cuda/optix/optix301/sample1manual/optixsample1.bash && optixsample1-env $* ; }
assimp-(){      . $(env-home)/graphics/assimp/assimp.bash && assimp-env $* ; }
assimptest-(){      . $(env-home)/graphics/assimp/AssimpTest/assimptest.bash && assimptest-env $* ; }
optixtest-(){      . $(env-home)/optix/OptiXTest/optixtest.bash && optixtest-env $* ; }
raytrace-(){      . $(env-home)/graphics/raytrace/raytrace.bash && raytrace-env $* ; }
mercurial-(){      . $(env-home)/hg/mercurial.bash && mercurial-env $* ; }
virtualgl-(){      . $(env-home)/graphics/virtualgl/virtualgl.bash && virtualgl-env $* ; }
nvidia-(){      . $(env-home)/graphics/nvidia/nvidia.bash && nvidia-env $* ; }
cudaz-(){      . $(env-home)/cuda/cudaz/cudaz.bash && cudaz-env $* ; }
mesa-(){      . $(env-home)/graphics/opengl/mesa/mesa.bash && mesa-env $* ; }
libpng-(){      . $(env-home)/graphics/libpng/libpng.bash && libpng-env $* ; }
macrosim-(){      . $(env-home)/optix/macrosim/macrosim.bash && macrosim-env $* ; }
assimpwrap-(){      . $(env-home)/graphics/assimpwrap/assimpwrap.bash && assimpwrap-env $* ; }
goofit-(){      . $(env-home)/cuda/goofit/goofit.bash && goofit-env $* ; }
vmd-(){      . $(env-home)/optix/vmd/vmd.bash && vmd-env $* ; }
openrl-(){      . $(env-home)/graphics/openrl/openrl.bash && openrl-env $* ; }
ggeo-(){      . $(env-home)/optix/ggeo/ggeo.bash && ggeo-env $* ; }
cudatex-(){      . $(env-home)/cuda/texture/cudatex.bash && cudatex-env $* ; }
optixtex-(){      . $(env-home)/optix/optixtex/optixtex.bash && optixtex-env $* ; }
unity-(){      . $(env-home)/graphics/unity/unity.bash && unity-env $* ; }
pbs-(){      . $(env-home)/graphics/shading/pbs.bash && pbs-env $* ; }
cudawrap-(){    . $(env-home)/cuda/cudawrap/cudawrap.bash && cudawrap-env $* ; }
optixsamples-(){      . $(env-home)/optix/optixsamples.bash && optixsamples-env $* ; }
iray-(){      . $(env-home)/iray/iray.bash && iray-env $* ; }
glfw-(){      . $(env-home)/graphics/glfw/glfw.bash && glfw-env $* ; }
glew-(){      . $(env-home)/graphics/glew/glew.bash && glew-env $* ; }
glfwtest-(){      . $(env-home)/graphics/glfw/glfwtest/glfwtest.bash && glfwtest-env $* ; }
vl-(){      . $(env-home)/graphics/vl/vl.bash && vl-env $* ; }
vltest-(){      . $(env-home)/graphics/vl/vltest/vltest.bash && vltest-env $* ; }
oglplus-(){      . $(env-home)/graphics/oglplus/oglplus.bash && oglplus-env $* ; }
oglplustest-(){      . $(env-home)/graphics/oglplus/oglplustest/oglplustest.bash && oglplustest-env $* ; }
oglrap-(){      . $(env-home)/graphics/oglrap/oglrap.bash && oglrap-env $* ; }
ggeoview-(){      . $(env-home)/graphics/ggeoview/ggeoview.bash && ggeoview-env $* ; }
glm-(){      . $(env-home)/graphics/glm/glm.bash && glm-env $* ; }
gl-(){      . $(env-home)/graphics/opengl/gl.bash && gl-env $* ; }
wendy-(){      . $(env-home)/graphics/wendy/wendy.bash && wendy-env $* ; }
basio-(){      . $(env-home)/boost/basio/basio.bash && basio-env $* ; }
asio-(){      . $(env-home)/network/asio/asio.bash && asio-env $* ; }
numpyserver-(){  . $(env-home)/boost/basio/numpyserver/numpyserver.bash && numpyserver-env $* ; }
photonio-(){      . $(env-home)/graphics/photonio/photonio.bash && photonio-env $* ; }
fishtank-(){      . $(env-home)/graphics/fishtank/fishtank.bash && fishtank-env $* ; }
sdl-(){      . $(env-home)/graphics/sdl/sdl.bash && sdl-env $* ; }
sfml-(){      . $(env-home)/graphics/sfml/sfml.bash && sfml-env $* ; }
gleq-(){      . $(env-home)/graphics/gleq/gleq.bash && gleq-env $* ; }
gleqtest-(){  . $(env-home)/graphics/glfw/gleqtest/gleqtest.bash && gleqtest-env $* ; }
bpo-(){      . $(env-home)/boost/bpo/bpo.bash && bpo-env $* ; }
asiozmq-(){      . $(env-home)/network/asiozmq/asiozmq.bash && asiozmq-env $* ; }
asiozmqtest-(){      . $(env-home)/network/asiozmqtest/asiozmqtest.bash && asiozmqtest-env $* ; }
azmq-(){      . $(env-home)/network/azmq/azmq.bash && azmq-env $* ; }
asiosamples-(){      . $(env-home)/boost/basio/asiosamples/asiosamples.bash && asiosamples-env $* ; }
bcfg-(){      . $(env-home)/boost/bpo/bcfg/bcfg.bash && bcfg-env $* ; }
bcfgtest-(){  . $(env-home)/boost/bpo/bcfg/test/bcfgtest.bash && bcfgtest-env $* ; }
optixrap-(){      . $(env-home)/graphics/optixrap/optixrap.bash && optixrap-env $* ; }
hrt-(){      . $(env-home)/graphics/hybrid-rendering-thesis/hrt.bash && hrt-env $* ; }
ppm-(){      . $(env-home)/graphics/ppm/ppm.bash && ppm-env $* ; }
ppmfast-(){      . $(env-home)/graphics/ppmfast/ppmfast.bash && ppmfast-env $* ; }
blogg-(){      . $(env-home)/boost/blogg/blogg.bash && blogg-env $* ; }
ntuwireless-(){      . $(env-home)/admin/ntuwireless.bash && ntuwireless-env $* ; }
npy-(){      . $(env-home)/numerics/npy/npy.bash && npy-env $* ; }
bfs-(){      . $(env-home)/boost/bfs/bfs.bash && bfs-env $* ; }
bpt-(){      . $(env-home)/boost/bpt/bpt.bash && bpt-env $* ; }
word-(){      . $(env-home)/tools/word/word.bash && word-env $* ; }
pages-(){      . $(env-home)/tools/pages/pages.bash && pages-env $* ; }
docx-(){      . $(env-home)/tools/docx/docx.bash && docx-env $* ; }
docxbuilder-(){      . $(env-home)/tools/docxbuilder/docxbuilder.bash && docxbuilder-env $* ; }
mono-(){      . $(env-home)/tools/mono/mono.bash && mono-env $* ; }
openxml-(){      . $(env-home)/tools/openxml/openxml.bash && openxml-env $* ; }
argparse-(){      . $(env-home)/python/argparse/argparse.bash && argparse-env $* ; }
hgssh-(){      . $(env-home)/hg/hgssh.bash && hgssh-env $* ; }
imagecapture-(){      . $(env-home)/osx/imagecapture.bash && imagecapture-env $* ; }
preview-(){      . $(env-home)/osx/preview/preview.bash && preview-env $* ; }
brandom-(){      . $(env-home)/boost/random/brandom.bash && brandom-env $* ; }
librocket-(){      . $(env-home)/graphics/gui/librocket/librocket.bash && librocket-env $* ; }
imgui-(){      . $(env-home)/graphics/gui/imgui/imgui.bash && imgui-env $* ; }
imguitest-(){      . $(env-home)/graphics/gui/imguitest/imguitest.bash && imguitest-env $* ; }
