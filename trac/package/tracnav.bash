
tracnav-usage(){

   package-usage ${FUNCNAME/-*/}
   cat << EOU
EOU

}

tracnav-env(){
   elocal-
   python-
   package-

  local branch
  case $(trac-major) in 
     0.10) branch=tracnav      ;;
     0.11) branch=tracnav-0.11 ;;
        *) echo $msg ABORT trac-major $(trac-major) not handled ;;
  esac
  export TRACNAV_BRANCH=$branch
}

tracnav-docurl(){   echo http:// ; }
tracnav-url(){      echo http://svn.ipd.uka.de/repos/javaparty/JP/trac/plugins/$(tracnav-obranch) ;}
tracnav-package(){  echo tracnav ; }

tracnav-fix(){
   local msg="=== $FUNCNAME :"
   echo $msg no fix needed
}



tracnav-obranch(){   package-obranch   ${FUNCNAME/-*/} $* ; }
tracnav-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
tracnav-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
tracnav-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
tracnav-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
tracnav-get(){       package-get       ${FUNCNAME/-*/} $* ; }

tracnav-install(){   package-install   ${FUNCNAME/-*/} $* ; }
tracnav-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
tracnav-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
tracnav-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

tracnav-status(){    package-status    ${FUNCNAME/-*/} $* ; }
tracnav-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
tracnav-diff(){      package-diff      ${FUNCNAME/-*/} $* ; } 
tracnav-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
tracnav-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }

tracnav-fullname(){  package-fullname  ${FUNCNAME/-*/} $* ; }