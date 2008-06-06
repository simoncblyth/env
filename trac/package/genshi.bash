genshi-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
   
     not needed for TRAC 0.10 
          ... but it would do no harm
            
     tags/0.4.4  ... does not work with TracTags ListTagged ... so use trunk
     trunk

   
     
EOU

}

genshi-env(){
  elocal-
  tpackage-
  
  local branch
  local tm=$(trac-major)
  case $tm in 
     0.10) branch=SKIP         ;;
     0.11) branch=trunk        ;;
        *) echo $msg ABORT trac-major $(trac-major) not handled ;;
  esac
  
  #echo $msg trac-major $tm branch $branch  
  export GENSHI_BRANCH=$branch
  
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



