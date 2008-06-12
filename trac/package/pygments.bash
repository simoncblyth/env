pygments-usage(){

   package-fn $FUNCNAME $*

cat << EOU

   pygments uses mercurial and is made available via easy_install
       
EOU

}


pygments-env(){
   elocal-
   package-
   
   export PYGMENTS_BRANCH=Pygments
}

pygments-url(){
   ## providing a url that does not start with http:// will cause an easy_install --editable -b . download
   echo $(pygments-branch)
}



pygments-branch(){    package-fn $FUNCNAME $* ; }
pygments-basename(){  package-fn $FUNCNAME $* ; }
pygments-dir(){       package-fn $FUNCNAME $* ; } 
pygments-egg(){       package-fn $FUNCNAME $* ; }
pygments-get(){       package-fn $FUNCNAME $* ; }

pygments-install(){   package-fn $FUNCNAME $* ; }
pygments-uninstall(){ package-fn $FUNCNAME $* ; }
pygments-reinstall(){ package-fn $FUNCNAME $* ; }
pygments-enable(){    package-fn $FUNCNAME $* ; }

pygments-status(){    package-fn $FUNCNAME $* ; }
pygments-auto(){      package-fn $FUNCNAME $* ; }
pygments-diff(){      package-fn $FUNCNAME $* ; } 
pygments-rev(){       package-fn $FUNCNAME $* ; } 
pygments-cd(){        package-fn $FUNCNAME $* ; }

pygments-fullname(){  package-fn $FUNCNAME $* ; }
pygments-update(){    package-fn $FUNCNAME $* ; }