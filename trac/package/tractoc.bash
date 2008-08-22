tractoc-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
   
    tractoc-fix :
        inplace edit to allow the top level headings to be listed
        without having to specify the page name in the [[TOC]]    
   
EOU

}

tractoc-env(){
  elocal-
  package-
  
  local branch
  case $(trac-major) in 
     0.10) branch=0.10 ;;
     0.11) branch=0.11 ;;
        *) echo $msg ABORT trac-major $(trac-major) not handled ;;
  esac
  export TRACTOC_BRANCH=$branch

}

tractoc-revision(){ echo 3783 ; }
tractoc-url(){     echo http://trac-hacks.org/svn/tocmacro/$(tractoc-branch) ;}
tractoc-package(){ echo tractoc ; }

tractoc-fix(){
   local msg="=== $FUNCNAME :"
   cd $(tractoc-dir)   
  # perl -pi -e 's/(min_depth.*)2(\s*# Skip.*)/${1}1${2} fixed by tractoc-fix/' tractoc/macro.py
  # svn diff tractoc/macro.py
  echo $msg now done with the auto patch 

}




tractoc-makepatch(){  package-fn $FUNCNAME $* ; }
tractoc-applypatch(){ package-fn $FUNCNAME $* ; }


tractoc-branch(){    package-fn  $FUNCNAME $* ; }
tractoc-basename(){  package-fn  $FUNCNAME $* ; }
tractoc-dir(){       package-fn  $FUNCNAME $* ; }  
tractoc-egg(){       package-fn  $FUNCNAME $* ; }
tractoc-get(){       package-fn  $FUNCNAME $* ; }    

tractoc-install(){   package-fn  $FUNCNAME $* ; }
tractoc-uninstall(){ package-fn  $FUNCNAME $* ; } 
tractoc-reinstall(){ package-fn  $FUNCNAME $* ; }
tractoc-enable(){    package-fn  $FUNCNAME $* ; }  

tractoc-status(){    package-fn  $FUNCNAME $* ; } 
tractoc-auto(){      package-fn  $FUNCNAME $* ; } 
tractoc-diff(){      package-fn  $FUNCNAME $* ; } 
tractoc-rev(){       package-fn  $FUNCNAME $* ; } 
tractoc-cd(){        package-fn  $FUNCNAME $* ; } 

tractoc-fullname(){  package-fn  $FUNCNAME $* ; } 
tractoc-update(){    package-fn  $FUNCNAME $* ; } 







