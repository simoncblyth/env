#!/bin/bash
#
#   CMT package test runner 
#
#   Usage : 
#         runtest.sh <pkgname> [<masterpath>]
#
#   It must be invoked from a NuWa-<release> dir with for example : 
#        cd NuWa-trunk
#        env -i path/to/runtest.sh rootiotest 
#        env -i path/to/runtest.sh rootiotest /
#
#   Alternative environment isolation using shim script : isotest.sh
#        cd NuWa-trunk
#        path/to/isotest.sh rootiotest
#
#   Arguments:
#         pkgname   :  case insensitive pkg name to be tested, eg rootiotest
#
#      masterpath   :  build triggering repository path (usually "/")
#                      this is only needed when creating xml reports of test results
#                      for the master trac instance    
#
#   Developed with the  "slv-" functions which provide combined
#   recipe generation and testing of the recipes and this script.
#
#
#
## first path relative to the second 
rt-relativeto(){ [ "$1" == "$2" -o "$2" == "" ] && echo "" || echo ${1/$2\//} ; }

## lowercase
rt-lower(){     echo $1 | tr "[A-Z]" "[a-z]" ; }

## cd into pkg directory and cmt environment 
rt-cdpkg(){
  local pkd=$1
  set --
  [ ! -d "$pkd/cmt" ] && echo $msg ERROR no dir $pkd/cmt && return 1
  cd $pkd/cmt
  [ ! -f setup.sh ] && cmt config
  . setup.sh 
  cd $pkd 
}

## pkgdir from the name ... case insensitive
rt-pkgdir(){
  local pkg=${1:-RootIOTest}
  local var=$(echo $pkg | tr "[a-z]" "[A-Z]")ROOT
  eval local dir=\$$var
  echo $dir  
}

## repository relative path from working copy dir
rt-reporoot(){ LANG=C svn info ${1:-$PWD} | perl -n -e 's,Repository Root: (\S*),$1, && print' - ; }
rt-repopath(){ LANG=C svn info ${1:-$PWD} | perl -n -e "s,URL: $(rt-reporoot)/(\S*),\$1, && print" - ; }

rt-main(){
   local pkg=$1        ; shift
   local masterpath=$1 ; shift

   export RT_PKG=$pkg
   export RT_OPTS=$*
   export RT_HOME=$PWD    ## invoking PWD of name ~ "NuWa-<release>"
 
   unset SITEROOT 
   unset CMSPROJECTPATH 
   unset CMTPATH 
   unset CMTEXTRATAGS 
   unset CMTCONFIG 

   . $RT_HOME/setup.sh  
   rt-cdpkg $RT_HOME/dybgaudi/DybRelease 

   ## get into environment and directory for pkg
   local pkgdir=$(rt-pkgdir $pkg)
   if [ -d "$pkgdir" ]; then 
      rt-cdpkg $pkgdir
   else
      export RT_ERROR="$msg ERROR no pkgdir for $pkg" 
   fi

   if [ -z "$masterpath" ]; then  
      export RT_COMMAND="nosetests $opts "
   else
      ## bitten xml reports needs repo paths relative to the build triggering MASTERPATH
      export RT_REPOPATH=$(rt-repopath $pkgdir)
      export RT_MASTERPATH=${masterpath:-/}
      export RT_BASEPREFIX=$(rt-relativeto $RT_REPOPATH $RT_MASTERPATH)/ 
      export RT_XMLOUT=$RT_HOME/../test-$(rt-lower $pkg).xml 
      export RT_COMMAND="nosetests $opts --with-xml-output --xml-outfile=$RT_XMLOUT --xml-baseprefix=$RT_BASEPREFIX"
   fi

   env | grep RT_ | sort 
   [ -z "$RT_ERROR" ] && eval $RT_COMMAND || echo $RT_ERROR
}
rt-main $*
