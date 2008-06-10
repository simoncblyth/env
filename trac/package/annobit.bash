
annobit-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
 
   
EOU

}

annobit-env(){
  elocal-
  package-
  export ANNOBIT_BRANCH=trunk
}

annobit-url(){     echo http://dayabay.phys.ntu.edu.tw/repos/tracdev/annobit/$(annobit-obranch) ;}
annobit-package(){ echo annobit ; }

annobit-fix(){
   cd $(annobit-dir)   
   echo no fix
}


annobit-obranch(){   package-obranch   ${FUNCNAME/-*/} $* ; }
annobit-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
annobit-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
annobit-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
annobit-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
annobit-get(){       package-get       ${FUNCNAME/-*/} $* ; }

annobit-install(){   package-install   ${FUNCNAME/-*/} $* ; }
annobit-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
annobit-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
annobit-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

annobit-status(){    package-status    ${FUNCNAME/-*/} $* ; }
annobit-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
annobit-diff(){      package-diff      ${FUNCNAME/-*/} $* ; } 
annobit-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
annobit-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }

annobit-fullname(){  package-fullname  ${FUNCNAME/-*/} $* ; }


annobit-prepare(){

   annobit-enable $*
  

}


