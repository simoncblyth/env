package-env(){
  elocal-
  python-
}


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




package-usage(){
    local name=$1
    local bn=$(package-branchname $name)
    local ob=$($name-obranch)
    
cat << EOU


    ENHANCEMENT IDEA ...  
       allow auto sensing that customizations are done and not propagaated

    See examples of usage of this in trac/package/ with precursors defined 
    in trac/trac.bash


    Functions of the branch ... 
   
    $name-branch    : $($name-branch)     formerly : used branch names ending _cust for  local customizing
    $name-obranch   : $($name-obranch)    formerly : with _cust stripped
                                          these are always the same now though
    
    $name-basename  : $($name-basename)   name of the leaf folder of the branch
    $name-url       : $($name-url)
    $name-dir       : $($name-dir)


    $name-fullname  : $($name-fullname)
       <name>-<version> string from setup.py --fullname , used to predict the egg name
    
    
    $name-egg-deprecated  : $(package-egg-deprecated $name)
        name of the egg, NB if there are native extensions you will need to append to this in an override 
        TODO : glean native egg names from the setup to avoid the need to override 
     
    $name-egg          :   $($name-egg)
         gleaned using \$ENV_HOME/python/pkgmeta.py examination of setup.py
         ... which works correctly even with native eggs ... but not very quick 
          NOPE .. still have to override and append to get native eggs correct
          
          
    $name-get       :
          svn co the $($name-url) into $($name-dir)  

    $name-install  :
          invoke $name-cust then 
          easy install into PYTHON_SITE $PYTHON_SITE
           
    $name-uninstall :
           remove the $($name-egg) and easy-install.pth reference
           
    $name-reinstall :
        uninstall then reinstall ... eg to propagate customizations 


    $name-reldir    : $($name-reldir)

    package-odir-  <name>

         the dir into which the checkout is done ... normally also the one with the setup.py 
         unless a non blank $name-reldir is defined 



    Usage :
        tpackage-
        $name-
        $name-usage

     Get a branch ready for customization  : $bn=${ob}_cust $name-get
     Check the dynamics                    : $bn=${ob}_cust $name-usage
     Install the default cust version      : $bn=${ob}_cust $name-install
       
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
   echo $msg fn $fn pn $pn cn $cn cal [$cal]
   eval $cal
   
}



package-diff(){
  local name=$1
  local dir=$($name-dir)
  svn diff $dir 
}

package-rev(){
  local name=$1
  local dir=$($name-dir)
  svnversion $dir 
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
   package-status- $name
   local s=$?
   local act=$(package-action- $s) 
   
   case $act in 
     skip) echo $msg $name ===\> $act ... nothing to do             ;;
      get) $name-get     ;;
  install) $name-install ;;  
    abort) echo $msg $name ===\> $act ... ABORTING && sleep 10000000 ;;
        *) echo $msg $name ===\> $act ... ERROR act not handled && sleep 10000000 ;;
   esac
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
  esac
}

package-status--(){
  case $1 in 
    0) echo installed ;;
    1) echo not downloaded ;;
    2) echo not installed  ;;
    3) echo the egg is not a directory ... non standard installation used ... delete and try again ;;
    4) echo branch is SKIP, this package is not needed for this trac version ;; 
  esac 
}


package-smry(){
  local name=$1
  local egg=$($name-egg)
  local branch=$($name-branch)
  local dir=$($name-dir)
  printf "%-15s %-40s %-50s %-70s" $name $branch $egg $dir
}

package-status-(){

  local msg="=== $FUNCNAME :"
  local name=$1
  
  local branch=$($name-branch)
  [ "$branch" == "SKIP" ] && return 4
  
  local dir=$($name-dir)
  #echo $msg $name $egg $dir
  
  [ ! -d $dir ] && return 1 
  
  local egg=$PYTHON_SITE/$($name-egg)
  [ -f $egg   ] && return 3 
  [ ! -d $egg ] && return 2
  return 0
}



package-odir-(){

  ## the dir into which the checkout is done
  ## ... normally also the one with the setup.py 
  ## unless $name-reldir is defined 
  
  local name=$1
  local base=$(package-basename $name)
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
   echo $(basename $branch)
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

package-egg(){
   local name=$1
   local dir=$($name-dir)
   local iwd=$PWD
   
   ## curios seems necessary to cd there .. cannot do from afar ??
   cd $dir
   python $ENV_HOME/python/pkgmeta.py setup.py
   cd $iwd  
}

package-egg-deprecated(){

   local name=$1
   local full=$($name-fullname)
   echo $full-py$PYTHON_MAJOR.egg
}



package-get(){
   local msg="=== $FUNCNAME :"
   local name=$1
   
   local odir=$(package-odir- $name)
   local dir=$($name-dir)
   
   [ "$odir" != "$dir" ] && echo $msg WARNING using relative dir shift odir $odir dir $dir 
   
   local url=$($name-url)
   local pir=$(dirname $odir)
   local bnm=$(basename $odir) 
   mkdir -p $pir
   cd $pir   
   
   echo $msg checkout $url into $pir with basename $bnm
   svn co $url $bnm
   
   package-look-version $bnm
}

package-look-version(){

   ## deprecated... as not doing _cust fiddling any more  

   local msg="=== $FUNCNAME :"
   local dir=$1
   local setup=$dir/setup.py
   local vers=$(python $setup --version)
   
   if [ -f $setup ]; then
      echo $msg version in the setup $setup $vers
   else
      echo $msg WARNING no setup $setup 
   fi
}






package-install(){
   local name=$1
   
   local msg="=== $FUNCNAME :"
   echo $msg $name 
   
   $name-cust
   $name-fix
   
   local dir=$($name-dir)
   cd $dir
   $SUDO easy_install -Z .   
   
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

package-obranch(){
   local name=$1
   local b=$($name-branch)
   [ "${b:$((${#b}-5))}" == "_cust" ] && b=${b/_cust/} 
   echo $b
}






package-setup-cust-deprecated(){

    local name=$1
    local msg="=== $FUNCNAME :"
    local dir=$($name-dir)
    local ver=$($name-setver)
    
    
    echo $msg $dir editing setup.py to change $ver to cust_$ver
   
    local iwd=$PWD   
    cd $dir
    [ ! -f setup.py ] && echo $msg ERROR setup.py in $dir not found && return 1 
    
    ## start with repo original in case of rerunning  
    svn revert setup.py
    perl -pi -e "s/(version\s*=\s*)(.)($ver)(.)(.*)$/\$1\$2cust_\$3\$4\$5/" setup.py
    svn diff setup.py

    cd $iwd

}

package-cust-deprecated(){

   local msg="=== $FUNCNAME :"
   local name=$1
   echo $msg $name 
   local base=$($name-basename)
   package-is-cust $base &&  package-setup-cust $name
}


package-is-cust-deprecated(){
    local b=$1
    [ "${b:$((${#b}-5))}" == "_cust" ]  && return 0 || return 1
}

package-egg-deprecated(){

  
   local name=$1
   local branch=$($name-branch)
   
   local eggver=$($name-eggver)
   local eggbas=$($name-eggbas)
    
   package-is-cust $branch && eggver=cust_${eggver}
   echo $eggbas-$eggver-py2.5.egg
 
}








