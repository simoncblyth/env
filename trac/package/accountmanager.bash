accountmanager-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
   
   Complication in configuration due to need to coordinate with :
      \$TRAC_APACHE2_CONF       : $TRAC_APACHE2_CONF
      \$APACHE2_LOCAL/trac.conf : $APACH2_LOCAL/trac.conf
   
EOU

}

accountmanager-env(){
  elocal-
  package-
  
  local branch
  case $(trac-major) in 
     0.10) branch=0.10 ;;
     0.11) branch=trunk ;;
        *) echo $msg ABORT trac-major $(trac-major) not handled ;;
  esac
  
  export ACCOUNTMANAGER_BRANCH=$branch
}

accountmanager-docurl(){  echo http://trac-hacks.org/wiki/AccountManagerPlugin ;}
accountmanager-url(){     echo http://trac-hacks.org/svn/accountmanagerplugin/$(accountmanager-branch) ;}
accountmanager-pkgname(){ echo acct_mgr ; }

#accountmanager-revision(){ echo 3981 ; }
accountmanager-revision(){  echo 4117 ; }


accountmanager-fix(){
   cd $(accountmanager-dir)   
   echo no fixes
}



accountmanager-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
accountmanager-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
accountmanager-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
accountmanager-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
accountmanager-get(){       package-get       ${FUNCNAME/-*/} $* ; }

accountmanager-install(){   package-install   ${FUNCNAME/-*/} $* ; }
accountmanager-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
accountmanager-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
accountmanager-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

accountmanager-status(){    package-status    ${FUNCNAME/-*/} $* ; }
accountmanager-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
accountmanager-diff(){      package-diff      ${FUNCNAME/-*/} $* ; } 
accountmanager-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
accountmanager-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }

accountmanager-fullname(){  package-fullname  ${FUNCNAME/-*/} $* ; }





