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

tractoc-url(){     echo http://trac-hacks.org/svn/tocmacro/$(tractoc-obranch) ;}
tractoc-package(){ echo tractoc ; }

tractoc-fix(){
   cd $(tractoc-dir)   
   perl -pi -e 's/(min_depth.*)2(\s*# Skip.*)/${1}1${2} fixed by tractoc-fix/' tractoc/macro.py
   svn diff tractoc/macro.py
}


tractoc-obranch(){   package-obranch   ${FUNCNAME/-*/} $* ; }
tractoc-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
tractoc-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
tractoc-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
tractoc-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
tractoc-get(){       package-get       ${FUNCNAME/-*/} $* ; }

tractoc-install(){   package-install   ${FUNCNAME/-*/} $* ; }
tractoc-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
tractoc-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
tractoc-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

tractoc-status(){    package-status    ${FUNCNAME/-*/} $* ; }
tractoc-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
tractoc-diff(){      package-diff      ${FUNCNAME/-*/} $* ; } 
tractoc-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
tractoc-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }

tractoc-fullname(){  package-fullname  ${FUNCNAME/-*/} $* ; }







