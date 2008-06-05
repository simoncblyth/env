tractrac-usage(){
   plugins-usage  ${FUNCNAME/-*/}
   cat << EOU
   
     
EOU

}

tractrac-env(){
  elocal-
  tplugins-
  
  #export TRACTRAC_BRANCH=tags/trac-0.10.4 
  #export TRACTRAC_BRANCH=tags/trac-0.11b1
  export TRACTRAC_BRANCH=tags/trac-0.11rc1
  #export TRACTRAC_BRANCH=trunk

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


tractrac-obranch(){   plugins-obranch   ${FUNCNAME/-*/} $* ; }
tractrac-branch(){    plugins-branch    ${FUNCNAME/-*/} $* ; }
tractrac-basename(){  plugins-basename  ${FUNCNAME/-*/} $* ; }
tractrac-dir(){       plugins-dir       ${FUNCNAME/-*/} $* ; } 
tractrac-egg(){       plugins-egg       ${FUNCNAME/-*/} $* ; }
tractrac-get(){       plugins-get       ${FUNCNAME/-*/} $* ; }
tractrac-cust(){      plugins-cust      ${FUNCNAME/-*/} $* ; }
tractrac-install(){   plugins-install   ${FUNCNAME/-*/} $* ; }
tractrac-uninstall(){ plugins-uninstall ${FUNCNAME/-*/} $* ; }
tractrac-reinstall(){ plugins-reinstall ${FUNCNAME/-*/} $* ; }
tractrac-enable(){    plugins-enable    ${FUNCNAME/-*/} $* ; }

tractrac-status(){    plugins-status    ${FUNCNAME/-*/} $* ; }
tractrac-auto(){      plugins-auto      ${FUNCNAME/-*/} $* ; }
tractrac-diff(){      plugins-diff      ${FUNCNAME/-*/} $* ; } 


