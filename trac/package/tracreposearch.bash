

tracreposearch-usage(){
   package-usage tracreposearch
   cat << EOU
  
      http://www.trac-hacks.org/wiki/RepoSearchPlugin
  
  
EOU

}

tracreposearch-env(){
   elocal-
   package-
   
   trac- 
    
  local branch
  case $(trac-major) in 
     0.11) branch=0.11     ;;
        *) echo $msg ABORT trac-major $(trac-major) not handled ;;
  esac
  export TRACREPOSEARCH_BRANCH=$branch
   
}


tracreposearch-revision(){ echo 4139 ; }
tracreposearch-url(){     echo http://trac-hacks.org/svn/reposearchplugin/$(tracreposearch-branch) ; }
tracreposearch-package(){ echo tracreposearch ; }

tracreposearch-fix(){
  local msg="=== $FUNCNAME :"
  echo $msg ... no fix 
}

tracreposearch-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
tracreposearch-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
tracreposearch-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
tracreposearch-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
tracreposearch-get(){       package-get       ${FUNCNAME/-*/} $* ; }

tracreposearch-install(){   package-install   ${FUNCNAME/-*/} $* ; }
tracreposearch-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
tracreposearch-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
tracreposearch-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

tracreposearch-status(){    package-status    ${FUNCNAME/-*/} $* ; }
tracreposearch-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
tracreposearch-diff(){      package-diff      ${FUNCNAME/-*/} $* ; }
tracreposearch-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
tracreposearch-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }

tracreposearch-fullname(){  package-fullname  ${FUNCNAME/-*/} $* ; }

tracreposearch-unconf(){

   local msg="=== $FUNCNAME :"
   local name=${1:-$SCM_TRAC}
   local tini=$SCM_FOLD/tracs/$name/conf/trac.ini
   local ver=$(basename $TRACTAGS_BRANCH)
   
   if [ "$ver" == "0.6" -o "$ver" == "trunk" ]; then
      echo $msg this is only relevant to pre 0.6 versions
   else
      trac-ini-
      trac-ini-edit $tini trac:default_handler:WikiModule
   fi

}

tracreposearch-conf(){
  
   local msg="=== $FUNCNAME :"
   trac-configure repo-search:include:\*.py:\*.h:\*.cxx:\*.xml repo-search:exclude:\*.png


}












tracreposearch-permission(){

   trac-admin- permission add authenticated REPO_SEARCH
   trac-admin- permission list
   
}