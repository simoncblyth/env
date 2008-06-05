genshi-usage(){
   plugins-usage  ${FUNCNAME/-*/}
   cat << EOU
   
     
EOU

}

genshi-env(){
  elocal-
  tplugins-
  
  export GENSHI_BRANCH=tags/0.4.4
 #export GENSHI_BRANCH=trunk

}

genshi-url(){     echo http://svn.edgewall.org/repos/genshi/$(genshi-obranch) ;}
genshi-package(){ echo genshi ; }
genshi-eggbas(){  echo Genshi ; }

genshi-eggver(){
    local ob=$(genshi-obranch)
    case $ob in 
        tags/0.4.4) echo 0.4.4 ;;
             trunk) echo 0.5   ;;     
                 *) echo $ob   ;;
    esac
}

genshi-fix(){
   cd $(genshi-dir)   
   echo no fixes
}


genshi-obranch(){   plugins-obranch   ${FUNCNAME/-*/} $* ; }
genshi-branch(){    plugins-branch    ${FUNCNAME/-*/} $* ; }
genshi-basename(){  plugins-basename  ${FUNCNAME/-*/} $* ; }
genshi-dir(){       plugins-dir       ${FUNCNAME/-*/} $* ; } 
genshi-egg(){       plugins-egg       ${FUNCNAME/-*/} $* ; }
genshi-get(){       plugins-get       ${FUNCNAME/-*/} $* ; }
genshi-cust(){      plugins-cust      ${FUNCNAME/-*/} $* ; }
genshi-install(){   plugins-install   ${FUNCNAME/-*/} $* ; }
genshi-uninstall(){ plugins-uninstall ${FUNCNAME/-*/} $* ; }
genshi-reinstall(){ plugins-reinstall ${FUNCNAME/-*/} $* ; }
genshi-enable(){    plugins-enable    ${FUNCNAME/-*/} $* ; }

genshi-status(){    plugins-status    ${FUNCNAME/-*/} $* ; }
genshi-auto(){      plugins-auto      ${FUNCNAME/-*/} $* ; }
genshi-diff(){      plugins-diff      ${FUNCNAME/-*/} $* ; } 




