
textile-usage(){

   package-fn $FUNCNAME $*

cat << EOU

    http://pypi.python.org/pypi/textile/                
       
EOU

}


textile-env(){
   elocal-
   package-
   export TEXTILE_BRANCH=textile-2.0.11.tar.gz
}

textile-url(){
   echo http://cheeseshop.python.org/packages/source/t/textile/$(textile-branch)
}




textile-branch(){    package-fn $FUNCNAME $* ; }
textile-basename(){  package-fn $FUNCNAME $* ; }
textile-dir(){       package-fn $FUNCNAME $* ; } 
textile-egg(){       package-fn $FUNCNAME $* ; }
textile-get(){       package-fn $FUNCNAME $* ; }

textile-install(){   package-fn $FUNCNAME $* ; }
textile-uninstall(){ package-fn $FUNCNAME $* ; }
textile-reinstall(){ package-fn $FUNCNAME $* ; }
textile-enable(){    package-fn $FUNCNAME $* ; }

textile-status(){    package-fn $FUNCNAME $* ; }
textile-auto(){      package-fn $FUNCNAME $* ; }
textile-diff(){      package-fn $FUNCNAME $* ; } 
textile-rev(){       package-fn $FUNCNAME $* ; } 
textile-cd(){        package-fn $FUNCNAME $* ; }

textile-fullname(){  package-fn $FUNCNAME $* ; }
textile-update(){    package-fn $FUNCNAME $* ; }

