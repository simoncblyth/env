package-env(){
  elocal-
  python-
}






package-usage(){
    local name=$1
    local bn=$(package-branchname $name)
    local ob=$($name-obranch)
    
cat << EOU
    Precursor "${name}-" is defined in trac/package/package.bash with precursor "tpackage-"

    Functions of the branch ... 
   
    $name-branch    : $($name-branch)    branch names ending _cust for  local customizing
    $name-obranch   : $($name-obranch)   with _cust stripped
    $name-basename  : $($name-basename)
    $name-url       : $($name-url)
    $name-dir       : $($name-dir)
    $name-eggver    : $($name-eggver)
       this is the egg version which must be manually edited to 
       match that from the original setup.py, this never has cust_ prepended
    
    $name-setver   : ...  mostly not yet implemented ...
       this is the version in the setup.py file .. its only needed
       when doing local cust 
    
    
    $name-egg       : $($name-egg)
        the version embedded in the name will have cust_ prepended if the 
        branch name ends with _cust
     
    $name-cust      :
          if the branch name ends in _cust then this attempts to edit the version in the setup.py,
          for this to work the $name-eggver function must have the original version   
        
    $name-get       :
          svn co the $($name-url) into $($name-dir)  

    $name-install  :
          invoke $name-cust then 
          easy install into PYTHON_SITE $PYTHON_SITE
           
    $name-uninstall :
           remove the $($name-egg) and easy-install.pth reference
           
    $name-reinstall :
        uninstall then reinstall ... eg to propagate customizations 

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
   echo $msg [$s] $(package-status-- $s) ===\> $a 
   return $s
}


package-action-(){
  case $1 in 
     0) echo skip ;;
     1) echo get  ;;
     2) echo install ;;
     3) echo abort ;;
  esac
}

package-status--(){
  case $1 in 
    0) echo installed ;;
    1) echo not downloaded ;;
    2) echo not installed  ;;
    3) echo the egg is not a directory ... non standard installation used ... delete and try again ;;
  esac 
}


package-status-(){

  local msg="=== $FUNCNAME :"
  local name=$1
  local egg=$PYTHON_SITE/$($name-egg)
  local dir=$($name-dir)
  echo $msg $name $egg $dir
  
  [ -f $egg   ] && return 3 
  [ ! -d $dir ] && return 1 
  [ ! -d $egg ] && return 2
  return 0
}


package-dir(){
  local n=$1
  local b=$(package-basename $n)
  echo $LOCAL_BASE/env/trac/package/$n/$b
}

package-basename(){
   local name=$1
   local branch=$($name-branch)
   echo $(basename $branch)
}


package-egg(){
   local name=$1
   local branch=$($name-branch)
   local eggver=$($name-eggver)
   local eggbas=$($name-eggbas)
   package-is-cust $branch && eggver=cust_${eggver}
   echo $eggbas-$eggver-py2.5.egg
}



package-get(){
   local msg="=== $FUNCNAME :"
   local name=$1
   local dir=$($name-dir)
   local url=$($name-url)
   local pir=$(dirname $dir)
   local bnm=$(basename $dir) 
   mkdir -p $pir
   cd $pir   
   
   echo $msg checkout $url into $pir with basename $bnm
   svn co $url $bnm
   
   package-look-version $bnm
}

package-look-version(){
   local msg="=== $FUNCNAME :"
   local dir=$1
   local setup=$dir/setup.py
   if [ -f $setup ]; then
      echo $msg looking for version in the setup $setup
      grep version $setup
      echo $msg this version should be placed into the -eggver function 
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
   local name=$1
   shift
   $($name-uninstall $*)
   $($name-install $*)
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


package-setup-cust(){

    local name=$1
    local msg="=== $FUNCNAME :"
    local dir=$($name-dir)
    local ver=$($name-setver)
    
    ## can setver be derived from the eggver ???
    
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


package-cust(){

   local msg="=== $FUNCNAME :"
   local name=$1
   echo $msg $name 
   local base=$($name-basename)
   package-is-cust $base &&  package-setup-cust $name
}


package-branch(){
   local bn=$(package-branchname $*) 
   local bv
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


package-is-cust(){
    local b=$1
    [ "${b:$((${#b}-5))}" == "_cust" ]  && return 0 || return 1
}








