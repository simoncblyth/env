xmlnose-usage(){
   package-fn  $FUNCNAME $*
   cat << EOU
EOU
}

xmlnose-env(){
  elocal-
  package-  
  export XMLNOSE_BRANCH=trunk/xmlnose
}

xmlnose-url(){     echo http://dayabay.phys.ntu.edu.tw/repos/env/$(xmlnose-branch) ; }
xmlnose-package(){ echo xmlnose ; }

xmlnose-fix(){
   cd $(xmlnose-dir)   
   echo no fixes
}

xmlnose-branch(){    package-fn $FUNCNAME $* ; }
xmlnose-basename(){  package-fn $FUNCNAME $* ; }
xmlnose-dir(){       package-fn $FUNCNAME $* ; } 
xmlnose-egg(){       package-fn $FUNCNAME $* ; }
xmlnose-get(){       package-fn $FUNCNAME $* ; }

xmlnose-install(){   package-fn $FUNCNAME $* ; }
xmlnose-uninstall(){ package-fn $FUNCNAME $* ; }
xmlnose-reinstall(){ package-fn $FUNCNAME $* ; }
xmlnose-enable(){    package-fn $FUNCNAME $* ; }

xmlnose-status(){    package-fn $FUNCNAME $* ; }
xmlnose-auto(){      package-fn $FUNCNAME $* ; }
xmlnose-diff(){      package-fn $FUNCNAME $* ; } 
xmlnose-rev(){       package-fn $FUNCNAME $* ; } 
xmlnose-cd(){        package-fn $FUNCNAME $* ; }

xmlnose-fullname(){  package-fn $FUNCNAME $* ; }
xmlnose-update(){    package-fn $FUNCNAME $* ; }