package-env(){
  elocal-
  python-
  #pkg-
  export PACKAGE_INSTALL_OPT=""
}


package-installed-(){ python -c "import $1" 2> /dev/null ; }


package-notes(){
cat << EON

  http://svn.python.org/projects/sandbox/branches/setuptools-0.6/
  http://svn.python.org/projects/sandbox/trunk/distutils_refactor/distutils/
  
  setup = distutils.core.setup
     which returns a Distribution object 

  http://svn.python.org/projects/sandbox/branches/setuptools-0.6/setuptools/command/bdist_egg.py

  
  class bdist_egg(Command)
      after finalize_options  this has egg_output   which is whats needed


  BINGO :
     http://svn.python.org/projects/distutils/trunk/misc/get_metadata.py

 
  


EON




}

package-pkgname(){
  local name=$1
  $name-pkgname 2> /dev/null || echo $name
}


package-revision(){
   local name=$1
   $name-revision 2> /dev/null || echo "" 
}


package-usage(){
    local name=$1
    local bn=$(package-branchname $name)

    
cat << EOU

    pkg.bash is the "for distribution" variant of this $BASH_SOURCE 
    that focusses on portability ... wherease this focusses on development convenience
    and is tied to a directory structure 


    ENHANCEMENT IDEA ...  
       allow auto sensing that customizations are done and not propagaated

    See examples of usage of this in trac/package/ with precursors defined 
    in trac/trac.bash


    Functions of the branch ... 
   
    $name-status    :
           if not installed by the package- machiney the egg info may be 
           incorrect ... workaround : uninstall and reinstall with the standard
           ez machinery with egg directories in the site-packages
           see ... python-ls 
   
   
    $name-branch    : $($name-branch)    
    $name-basename  : $($name-basename)   name of the leaf folder of the branch
          ... possibly should mangle the branch changing / to _ for example as it
          is possible that the leaf name is not sufficiently distinctive 
    $name-dir       : $($name-dir)
    
    \$(package-revision $name) : $(package-revision $name)
           if a particular revision is required ... in order to match a patch for example
           then implement a $name-revision method for the package 
    
    $name-url       : $($name-url)       
        the setup.py should be in the resulting checked out folder or if not a relative path from
        the checked out folder to the folder with the setup.py must be echoed by 
        an $name-reldir function  

    $name-fullname  : \$($name-fullname)
       <name>-<version> string from setup.py --fullname , used to predict the egg name
    
     
    $name-egg          :   \$($name-egg)
         gleaned using \$ENV_HOME/python/pkgmeta.py examination of setup.py
         BUT :
            tis getting the name wrong with dev and native eggs 
          
         ... but the reinstall still works 
               easy_install just replaces the prior egg ...   
           
    package-eggname-  <pkgname>
             determine the name of the egg from path reported by pkgname.__file__ 
    package-eggname <pkgname>
             invokes package-eggname- returning a blank in case of any error

           
           
           
    $name-get       :
          svn co the $($name-url) into $($name-dir)  

    $name-install  :
          invokes -applypatch which will do so if one exists 
          invoke $name-fix then 
          easy install into PYTHON_SITE $PYTHON_SITE
     
    package-applypatch $name
          patches are identified by the branch basename and checkout revision
          so following updates from upstream an appropriately named patch will not be found
          ... so to incoporate upstream mods do a manual svn update into the
          patched working copy and investigate problems/conflicts before making a 
          new patch              
                       
                                   
    PACKAGE_INSTALL_OPT=develop $name-install  :
          develop mode installation, allowing availability on sys.path to other
          packages direct from the source directories without creating eggs 
          
          Done with \$PYTHON_SITE/distname.egg-link containing the 
          directory path back to the source directory and putting this directory 
          into \$PYTHON_SITE/easy_install.pth also 
           ... this replaces existing normal installs appropriately,       
           
    
    $name-develop :
         a reimplementation 
                  
                                
    $name-uninstall :
           remove the \$($name-egg) and easy-install.pth reference
           
    $name-update    :
          invoke $name-get, $name-uninstall and $name-install

    $name-reinstall :
        uninstall then reinstall then restart apache2 ... eg to propagate customizations 
        TODO : make the restarting apache2 configurable ... tis only relevant to trac plugins
        
        NOTE ... on changing a patch the new patch fails to be applied with :
            "=== package-applypatch : ERROR there are local modifications ... cannot apply patch"
            workaround this by manually reverting before doing the reinstall
        
        
        
    package-odir-  <name>

         the dir into which the checkout is done ... normally also the one with the setup.py 
         unless a non blank $name-reldir is defined 

    package-initial-rev  <name>
         the checked out revision number as parsed from svn info 

    package-patchcmd  <name>
        the patch command to apply from a checked out folder, accounting for the right 
        number of slashes to be stripped from the svn diff patch file in the -pn

    package-pkgname <name>
        defers to $name-pkgname if defined to override the default of <name>
        allowing the cases where the script name does not corresponds to the 
        python packagename to be handled
        







    Usage :
        env-
        trac-
        $name-
        $name-usage

    To see the effect of changes...       : sudo apachectl restart

    python-ls :   list the eggs in \$PYTHON_SITE : $PYTHON_SITE
    python-pth :  cat the $PYTHON_SITE/easy-install.pth
    
    python-uninstall <eggname>
         manual uninstallation removing the egg dir and the easy-install.pth reference
          ... needed for to remove a prior version not "reachable" 
         by $name-egg 

    NB this flexibility is implemented by having everything that depends on $bn dynamic
        
EOU

}


package-fn(){

   local msg="=== $FUNCNAME :"
   
   local fnc=$1
   shift
   
   local pkg=${fnc/-*/}
   local cmd=${fnc/*-/}
   
   local cal="package-$cmd $pkg $*"
   #echo $msg fnc $fnc pkg $pkg cmd $cmd cal [$cal]
   eval $cal
   
}


package-patchpath(){
  local name=$1
  local basename=$($name-basename)
  local dir=$($name-dir)
  local irev=$(package-initial-rev $name) 
  local patchname=$name-$basename-$irev 
  local patchdir=$ENV_HOME/trac/patch/$name
  echo $patchdir/$patchname.patch  
}

package-makepatch(){
   local msg="=== $FUNCNAME :"
   local name=$1
   
   local pris=$(package-ispristine $name)
   
   if [ "$pris" == "not-svn" -o "$pris" == "pristine" ]; then
      echo $msg $name $pris so skip making patch ...
      return 
   fi
   
   
   local patchpath=$(package-patchpath $name)
   local patchdir=$(dirname $patchpath)
   mkdir -p $patchdir
   echo $msg writing \"svn diff\" to patchpath $patchpath ... remember to svn add and ci for safekeeping 
   package-diff $name > $patchpath  || echo $msg ERROR while creating patch  
   
}

package-ispristine-(){
   local name=$1
   local dir=$($name-dir)   
   [ ! -d "$dir/.svn" ] && return 3 
   [ "$(svn diff $dir)" = "" ] && return 0 || return 1   
}

package-ispristine-msg(){
    case $1 in 
       0) echo pristine    ;;
       1) echo local-mods  ;;
       3) echo not-svn ;;
       *) echo ERROR ;;
    esac  
}

package-ispristine(){
   package-ispristine- $*
   package-ispristine-msg $?
}


package-applypatch(){

   local msg="=== $FUNCNAME :"
   local name=$1
   local patchpath=$(package-patchpath $name)
   
   [ ! -f $patchpath ]    && echo $msg there is no patch file $patchpath && return 1 
   ! package-ispristine- $name && echo $msg ERROR there are local modifications ... cannot apply patch && return 1 


   local dir=$($name-dir)
   
   echo $msg applying patch $patchpath to pristine checkout from checkout folder $dir
   cd $dir
 
   local cmd=$(package-patchcmd $name)
   echo $cmd
   eval $cmd
   
   
   local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
   local chk=$tmp/check.patch
   svn diff > $chk
   
   echo $msg checking the svn diff after applying the patch with the patch ... should be just path context diffs
   diff $chk $patchpath
      
}

package-patchcmd-absolu(){
   
   ##
   ## this only gets the correct when the repo is at the same slash depth
   ##   may be could avoid the issue by a perl -pi on the svn diff output   
   ##   to make it relative
   ##
   ##
   ##   did like this because my svn diff were erroneously absolute
   ##
  
  
   local name=$1
   
   local dir=$($name-dir)
   ebash-
   local bsc=$(bash-slash-count $dir)
   local patchpath=$(package-patchpath $name)

   local pbsc=${2:-$bsc}

   local cmd="patch -p$bsc < $patchpath "
   echo $cmd

}

package-patchcmd(){

   local name=$1
   local dir=$($name-dir)
   local patchpath=$(package-patchpath $name)

   cd $dir
   local cmd="patch -p0 < $patchpath "
   echo $cmd

}


package-diff(){
  local msg="=== $FUNCNAME :"
  local name=$1
  local dir=$($name-dir)
  
  [ ! -d $dir/.svn ] && echo $msg $dir is not working copy && return 1
  
  local iwd=$PWD
  cd $dir
  svn diff 
  cd $iwd 
     
  ## duh ... do the diff from the dir to get relative paths    
     
}

package-rev(){
  local name=$1
  local dir=$($name-dir)
  svnversion $dir 
}

package-initial-rev(){
  local name=$1
  svn info  $($name-dir) | env -i perl -n -e 'm/^Revision: (\d*)/ && print $1 ' -
}


package-cd(){
  local name=$1
  local dir=$($name-dir)
  cd $dir 
}


package-auto(){

   local name=$1
   shift
   local msg="=== $FUNCNAME :"
   
   local steps="one two"
   for step in $steps
   do
       echo $msg $name step $step
       
       package-status- $name
       local s=$?
       local act=$(package-action- $s) 
   
       case $act in 
           skip) echo $msg $name ===\> $act ... nothing to do  && return 0  ;;
            get) $name-get     ;;
        install) $name-install ;;  
          abort) echo $msg $name ===\> $act ... ABORTING && sleep 10000000 ;;
              *) echo $msg $name ===\> $act ... ERROR act not handled && sleep 10000000 ;;
       esac
   done

}

package-status(){
   local msg="=== $FUNCNAME :"
   package-status- $*
   local s=$?
   local a=$(package-action- $s)
   printf "%130s %-20s %-20s\n" "$(package-smry $1)" "$(package-status-- $s)" "==> $a "
   return $s
}


package-action-(){
  case $1 in 
     0) echo skip ;;
     1) echo get  ;;
     2) echo install ;;
     3) echo abort ;;
     4) echo skip ;;
     5) echo install ;;
  esac
}

package-status--(){
  case $1 in 
    0) echo installed ;;
    1) echo not downloaded ;;
    2) echo not installed  ;;
    3) echo the egg is not a directory ... non standard installation used ... delete and try again ;;
    4) echo branch is SKIP, this package is not needed for this trac version ;; 
    5) echo eggname is blank , package not installed ?? ;;
  esac 
}

package-status-(){

  local msg="=== $FUNCNAME :"
  local name=$1
  
  local branch=$($name-branch)
  [ "$branch" == "SKIP" ] && return 4
  
  local dir=$($name-dir)
  [ ! -d $dir ] && return 1 
  
  local eggname=$($name-egg)
  #echo $msg name:$name eggname:$eggname dir:$dir
  
  [ "$eggname" == "" ] && return 5
  
  local egg=$PYTHON_SITE/$eggname
  
  [ -f $egg   ] && return 3 
  [ ! -d $egg ] && return 2
  return 0
}









package-egg(){
   local name=$1
   local pkgname=$(package-pkgname $name)
   local egg=$(package-eggname $pkgname)
   echo $egg
}





package-smry(){
  local name=$1
  #local egg=$($name-egg)
  local egg=$(package-egg $name)
  local branch=$($name-branch)
  local rev=$(package-revision $name)
  
  local pris=$(package-ispristine $name)
  printf "%-15s %-5s %-35s %-45s %-70s" $name ${rev:-HEAD} $branch ${egg:--} $pris
}

package-summary(){
  printf "%130s \n" "$(package-smry $1)"
}



package-odir-(){

  ## the dir into which the checkout is done
  ## ... normally also the one with the setup.py 
  ## unless $name-reldir is defined 
  
  local name=$1
  local base=$($name-basename)
  local dir=$LOCAL_BASE/env/trac/package/$name/$base
  echo $dir
}

package-dir(){
 
  local name=$1
  ## if a reldir is provided it should get from the 
  ## checked out folder to the folder with the setup.py 
  
  local dir=$(package-odir- $name)
  local reld=$($name-reldir 2> /dev/null || echo "" )
  
  [ "$reld" != "" ] && dir="$dir/$reld" 
  
  echo $dir
}

package-basename(){
   local name=$1
   local branch=$($name-branch)

   local bnm=$(basename $branch)   ## leaf of the branch 
   local  pkt=$(package-type $bnm)
   local tba=$(package-typebase $bnm $pkt)
   
   echo $tba
}


package-fullname(){
    local name=$1
    local dir=$($name-dir)
    local full
    local setup=$dir/setup.py

    if [ -f $setup ]; then
       full=$(python $setup --fullname)
    else
       full="no-setup-so-no-name-yet"
    fi
    echo $full
}



package-develop(){

    local msg="=== $FUNCNAME :"
    local name=$1
    local dir=$($name-dir)
    local iwd=$PWD

    cd $dir
    [ ! -f setup.py ] && echo "$msg no setup" && return 2 
    
    $SUDO python setup.py develop
   
    cd $iwd

}



package-eggname-(){    
   
   # python gives relative paths when done from the source directory containing the package 
   # ... hence the tmp move
   #
   #  hmmm when in development mode 
   #    bitten.__file__  points back to the source folder
   #      '/usr/local/env/trac/package/bitten/trac-0.11/bitten/__init__.pyc'
   #


python -c "$(cat << EOC
import os ; 
os.chdir('/tmp') ; 
import $1 as _ ; 
eggs=[egg for egg in _.__file__.split('/') if egg.endswith('.egg')] ; 
report = len(eggs) > 0 and eggs[0] or "no-egg" ;
print report
EOC)"
 
}



package-eggname(){
  package-eggname- $* 2> /dev/null || echo -n 
}



package-egg-deprecated(){
   local name=$1
   local dir=$($name-dir)
   local iwd=$PWD
   local setup=setup.py
   
   ## curios seems necessary to cd there .. cannot do from afar ??
   
   [ ! -d $dir ] && echo "" && return 1
   
   cd $dir
   [ ! -f $setup ] && echo "" && return 2 
   PYTHONPATH=. python $ENV_HOME/python/pkgmeta.py $setup
   local stat=$?
   cd $iwd  
   return $stat
}



package-type(){
   local b=$1
   if [ "${b:$((${#b}-7))}" == ".tar.gz" ]; then
      echo tar.gz
   elif [ "${b:$((${#b}-4))}" == ".zip" ]; then
      echo zip
   elif [ "${b:$((${#b}-4))}" == ".tgz" ]; then
      echo tgz
   else
      echo svn
   fi
}

package-typebase(){
   local b=$1
   local t=$2
   case $t in 
   tar.gz) echo ${b:0:$((${#b}-7))} ;;
      tgz) echo ${b:0:$((${#b}-4))} ;;
      zip) echo ${b:0:$((${#b}-4))} ;;
      svn) echo $b ;;
   esac
}


package-get(){
   local msg="=== $FUNCNAME :"
   local name=$1
   
   local odir=$(package-odir- $name)
   local dir=$($name-dir)
   
   [ "$odir" != "$dir" ] && echo $msg WARNING using relative dir shift odir $odir dir $dir 
   
   local url=$($name-url)
   local rev=$(package-revision $name)
   local pir=$(dirname $odir)
   
   local bnm=$(basename $odir) 
   mkdir -p $pir
   cd $pir   

   ## http://peak.telecommunity.com/DevCenter/EasyInstall#downloading-and-installing-a-package

   if [ "${url:0:7}" != "http://" -a "${url:0:6}" != "ftp://"  ]; then
       ## is this how easy_install -eb  always calls its downloads  ???
       local eznam=$(echo $bnm | tr "[A-Z]" "[a-z]")
       [ ! -d "$eznam" ] && easy_install --editable -b . $bnm || echo $msg already ez installed to $eznam 
        return
   fi



   ## need to original branchname to see the tgz/zip 
   local brn=$(basename $($name-branch))
   local  pkt=$(package-type $brn)
   local tba=$(package-typebase $brn $pkt)
   
   echo $msg brn:$brn bnm:$bnm pkt:$pkt tba:$tba url:$url



   
   if [ "$pkt" == "tgz" -o "$pkt" == "tar.gz" ]; then
   
      [ ! -f $brn ] && curl -L -O $url  || echo $msg already downloaded $brn 
      [ ! -d $tba ] && tar zxvf $brn || echo $msg already unpacked $tba
   
   elif [ "${b:$((${#b}-4))}" == ".zip" ]; then
     
      echo $msg caution unzip not tested
       
      [ ! -f $brn ] && curl -L -O $url
      [ ! -d $tba ] && unzip $brn 
       
   else 
   
      echo $msg svn checkout $url rev ${rev:-HEAD}   into $pir with basename $bnm
      svn co $url $bnm --revision ${rev:-HEAD}
      [ ! -d $bnm ] && echo $msg ABORT failed to checkout ... && sleep 10000000000
      
   fi
   
    package-look-version $bnm
  
}



package-look-version(){

   ## deprecated... as not doing _cust fiddling any more  

   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   local dir=$1
   local setup=$dir/setup.py
   
   if [ -f $setup ]; then
      cd $dir
      ## cannot use setup.py from afar 
      local vers=$(python setup.py --version) 
      cd $iwd
      echo $msg version in the setup $setup $vers
   else
      echo $msg WARNING no setup $setup 
   fi
}






package-install(){
   local name=$1
   
   local msg="=== $FUNCNAME :"
   echo $msg $name 
   
   
   package-applypatch $name 
   $name-fix 2> /dev/null
   
   local dir=$($name-dir)
   cd $dir

   if [ "$PACKAGE_INSTALL_OPT" == "develop" ]; then
      $SUDO python setup.py develop
   else
      $SUDO easy_install -Z .  
   fi
   
   # Note it is possible to install direct from a URL ... but 
   # that gives no opportunity for customization..
   # 
   #  easy_install -Z http://trac-hacks.org/svn/$macro/0.10/
   
}





package-uninstall(){
   local msg="=== $FUNCNAME :"
   local name=$1
   local egg=$($name-egg)
   echo $msg $name uninstalling egg $egg
   python-uninstall $egg
}

package-reinstall(){

   local msg="=== $FUNCNAME :"
   local name=$1
   shift
   
   PYTHON_UNINSTALL_DIRECTLY=yep $name-uninstall $*
   $name-install $*
   
   echo $msg restarting apache
   sudo apachectl restart
}


package-update(){

   local msg="=== $FUNCNAME :"
   local name=$1
   shift

   PYTHON_UNINSTALL_DIRECTLY=yep $name-uninstall $*
   
   $name-get $*
   $name-install $* 

}





package-enable(){

   local pame=$1
  
  ## NB the appropriate string is the python package name ...
  ##  test with  eg  python -c "import tractoc" 
  
   local pkgn=$($pame-package)
   trac-configure components:$pkgn.\*:enabled
}

package-test(){
    local pame=$1
    python -c "import $pame" 
}



package-branch(){
   local name=$1
   local bn=$(package-branchname $name) 
   local bv
   
   ## sometimes the NAME-env updates the NAME_BRANCH on the basis
   ## of TRAC_VERSION ... so must update here to feel that 
   ##
   ##  $name-env
   ##
   ## do not do that ... it prevents NAME_BRANCH being an input 
   ## ... the use case for 
   ##    TRAC_VERSION=0.10.4 name-blah 
   ##  is marginal anyhow 
   ##
   
   
   eval bv=\$$bn
   echo $bv
}

package-branchname(){
   local name=$1
   local NAME=$(echo $name | tr "[a-z]" "[A-Z]")
   local bn=${NAME}_BRANCH
   echo $bn
}













