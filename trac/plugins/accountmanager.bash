accountmanager-usage(){
   plugins-usage  ${FUNCNAME/-*/}
   cat << EOU
   
     
EOU

}

accountmanager-env(){
  elocal-
  tplugins-
  
 #export ACCOUNTMANAGER_BRANCH=0.10
  export ACCOUNTMANAGER_BRANCH=trunk

}


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


accountmanager-obranch(){   plugins-obranch   ${FUNCNAME/-*/} $* ; }
accountmanager-branch(){    plugins-branch    ${FUNCNAME/-*/} $* ; }
accountmanager-basename(){  plugins-basename  ${FUNCNAME/-*/} $* ; }
accountmanager-dir(){       plugins-dir       ${FUNCNAME/-*/} $* ; } 
accountmanager-egg(){       plugins-egg       ${FUNCNAME/-*/} $* ; }
accountmanager-get(){       plugins-get       ${FUNCNAME/-*/} $* ; }
accountmanager-cust(){      plugins-cust      ${FUNCNAME/-*/} $* ; }
accountmanager-install(){   plugins-install   ${FUNCNAME/-*/} $* ; }
accountmanager-uninstall(){ plugins-uninstall ${FUNCNAME/-*/} $* ; }
accountmanager-reinstall(){ plugins-reinstall ${FUNCNAME/-*/} $* ; }
accountmanager-enable(){    plugins-enable    ${FUNCNAME/-*/} $* ; }

accountmanager-status(){    plugins-status    ${FUNCNAME/-*/} $* ; }
accountmanager-auto(){      plugins-auto      ${FUNCNAME/-*/} $* ; }
accountmanager-diff(){      plugins-diff      ${FUNCNAME/-*/} $* ; } 
accountmanager-rev(){       plugins-rev       ${FUNCNAME/-*/} $* ; } 



