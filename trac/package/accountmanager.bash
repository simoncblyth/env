accountmanager-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
   
     
EOU

}

accountmanager-env(){
  elocal-
  tpackage-
  
 #export ACCOUNTMANAGER_BRANCH=0.10
  export ACCOUNTMANAGER_BRANCH=trunk

}

accountmanager-docurl(){  echo http://trac-hacks.org/wiki/AccountManagerPlugin ;}
accountmanager-url(){     echo http://trac-hacks.org/svn/accountmanagerplugin/$(accountmanager-obranch) ;}
accountmanager-package(){ echo accountmanager ; }
accountmanager-eggbas(){  echo TracAccountManager ; }

accountmanager-eggver(){
    local ob=$(accountmanager-obranch)
    case $ob in 
             trunk) echo 0.2.1dev_r3734   ;;     
                 *) echo $ob              ;;
    esac
}

accountmanager-fix(){
   cd $(accountmanager-dir)   
   echo no fixes
}


accountmanager-obranch(){   package-obranch   ${FUNCNAME/-*/} $* ; }
accountmanager-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
accountmanager-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
accountmanager-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
accountmanager-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
accountmanager-get(){       package-get       ${FUNCNAME/-*/} $* ; }
accountmanager-cust(){      package-cust      ${FUNCNAME/-*/} $* ; }
accountmanager-install(){   package-install   ${FUNCNAME/-*/} $* ; }
accountmanager-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
accountmanager-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
accountmanager-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

accountmanager-status(){    package-status    ${FUNCNAME/-*/} $* ; }
accountmanager-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
accountmanager-diff(){      package-diff      ${FUNCNAME/-*/} $* ; } 
accountmanager-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 



