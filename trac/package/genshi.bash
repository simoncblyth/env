genshi-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
   
 after installing genshi tags/0.4.4 ... ListTagged doesnt work   
   
   
     
EOU

}

genshi-env(){
  elocal-
  tpackage-
  
 #export GENSHI_BRANCH=tags/0.4.4
  export GENSHI_BRANCH=trunk

}

 
genshi-url(){     echo http://svn.edgewall.org/repos/genshi/$(genshi-obranch) ;}
genshi-package(){ echo genshi ; }
genshi-eggbas(){  echo Genshi ; }

genshi-eggver(){
    local ob=$(genshi-obranch)
    case $ob in 
        tags/0.4.4) echo 0.4.4 ;;
             trunk) echo 0.5dev_r858 ;;     
                 *) echo $ob   ;;
    esac
}

genshi-fix(){
   cd $(genshi-dir)   
   echo no fixes
}


genshi-obranch(){   package-obranch   ${FUNCNAME/-*/} $* ; }
genshi-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
genshi-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
genshi-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
genshi-egg(){       
     ## override needed due to native optimizations resulting in native egg name
     local egg=$(package-egg       ${FUNCNAME/-*/} $*) 
     case $NODE_TAG in 
        G) echo ${egg/.egg/}-macosx-10.5-ppc.egg ;;
        *) echo $egg ;;
     esac
}
genshi-get(){       package-get       ${FUNCNAME/-*/} $* ; }
genshi-cust(){      package-cust      ${FUNCNAME/-*/} $* ; }
genshi-install(){   package-install   ${FUNCNAME/-*/} $* ; }
genshi-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
genshi-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
genshi-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

genshi-status(){    package-status    ${FUNCNAME/-*/} $* ; }
genshi-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
genshi-diff(){      package-diff      ${FUNCNAME/-*/} $* ; } 
genshi-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
genshi-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }



