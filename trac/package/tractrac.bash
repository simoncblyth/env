tractrac-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
   
   
    addendum to tractrac-install
   
         In addition to installing the package into PYTHON_SITE this 
         also installs the trac-admin and tracd entry points by default 
         to /usr/local/bin/{trac-admin,tracd}  
     
EOU

}

tractrac-env(){
  elocal-
  tpackage-
  
  export TRACTRAC_BRANCH=$(tractrac-version2branch $TRAC_VERSION)
}


tractrac-version2branch(){
  local version=$1
  local branch
  case $version in 
      0.11b1) branch=tags/trac-0.11b1  ;;
     0.11rc1) branch=tags/trac-0.11rc1 ;; 
      0.10.4) branch=tags/trac-0.10.4  ;;
       trunk) branch=trunk ;;
  esac
  echo $branch
}

tractrac-url(){     echo http://svn.edgewall.org/repos/trac/$(tractrac-obranch) ;}
tractrac-package(){ echo trac ; }
tractrac-eggbas(){  echo Trac ; }

tractrac-eggver(){
    local ob=$(tractrac-obranch)
    case $ob in 
        tags/trac-0.10.4) echo 0.10.4 ;;
        tags/trac-0.11b1) echo 0.11b1   ;;
       tags/trac-0.11rc1) echo 0.11rc1  ;;
          tags/trac-0.11) echo 0.11     ;;
                       *) echo $ob      ;;
    esac
}

tractrac-fix(){
   cd $(tractrac-dir)   
   echo no fixes
}


tractrac-obranch(){   package-obranch   ${FUNCNAME/-*/} $* ; }
tractrac-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
tractrac-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
tractrac-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
tractrac-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
tractrac-get(){       package-get       ${FUNCNAME/-*/} $* ; }
tractrac-cust(){      package-cust      ${FUNCNAME/-*/} $* ; }
tractrac-install(){   package-install   ${FUNCNAME/-*/} $* ; }
tractrac-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
tractrac-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
tractrac-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

tractrac-status(){    package-status    ${FUNCNAME/-*/} $* ; }
tractrac-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
tractrac-diff(){      package-diff      ${FUNCNAME/-*/} $* ; } 
tractrac-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
tractrac-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }

