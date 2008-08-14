
bitextra-usage(){
   package-usage  ${FUNCNAME/-*/}
   cat << EOU
 
    Switch on the extras ... annotation and test summarizer with :
       SUDO=sudo bitextra-prepare
   

   
EOU

}

bitextra-env(){
  elocal-
  package-
  export BITEXTRA_BRANCH=trunk
}

bitextra-url(){     echo http://dayabay.phys.ntu.edu.tw/repos/tracdev/annobit/$(bitextra-branch) ;}


bitextra-fix(){
   cd $(bitextra-dir)   
   echo no fix
}


bitextra-package(){   echo bitextra ; }

bitextra-branch(){    package-branch    ${FUNCNAME/-*/} $* ; }
bitextra-basename(){  package-basename  ${FUNCNAME/-*/} $* ; }
bitextra-dir(){       package-dir       ${FUNCNAME/-*/} $* ; } 
bitextra-egg(){       package-egg       ${FUNCNAME/-*/} $* ; }
bitextra-get(){       package-get       ${FUNCNAME/-*/} $* ; }

bitextra-install(){   package-install   ${FUNCNAME/-*/} $* ; }
bitextra-uninstall(){ package-uninstall ${FUNCNAME/-*/} $* ; }
bitextra-reinstall(){ package-reinstall ${FUNCNAME/-*/} $* ; }
bitextra-enable(){    package-enable    ${FUNCNAME/-*/} $* ; }

bitextra-status(){    package-status    ${FUNCNAME/-*/} $* ; }
bitextra-auto(){      package-auto      ${FUNCNAME/-*/} $* ; }
bitextra-diff(){      package-diff      ${FUNCNAME/-*/} $* ; } 
bitextra-rev(){       package-rev       ${FUNCNAME/-*/} $* ; } 
bitextra-cd(){        package-cd        ${FUNCNAME/-*/} $* ; }

bitextra-fullname(){  package-fullname  ${FUNCNAME/-*/} $* ; }

bitextra-update(){    package-update    ${FUNCNAME/-*/} $* ; }


bitextra-prepare(){

   local name=${FUNCNAME/-*/}
   bitextra-enable $*
  
   ## trac-configure components:$name.TestResultsSummarizer:enabled  is included with the enable


   ## disable the default test summarizer in order for my modified one to take its place    
   
   trac-configure components:bitten.report.testing.TestResultsSummarizer:disabled

   ## verify http://localhost/tracs/workflow/admin/general/plugin
   ## to see that the swap is done OK 
   ##   ... NB you can reconfigure on the fly with trac 0.11 , no need to restart the instance


}


