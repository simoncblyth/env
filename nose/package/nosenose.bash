nosenose-usage(){
   package-fn  $FUNCNAME $*
   cat << EOU
EOU
}

nosenose-env(){
  elocal-
  package-  
  export NOSENOSE_BRANCH=tags/0.10.3-release
}

nosenose-url(){     echo http://python-nose.googlecode.com/svn/$(nosenose-branch) ; }
nosenose-package(){ echo nose ; }

nosenose-fix(){
   cd $(nosenose-dir)   
   echo no fixes
}


nosenose-branch(){    package-fn $FUNCNAME $* ; }
nosenose-basename(){  package-fn $FUNCNAME $* ; }
nosenose-dir(){       package-fn $FUNCNAME $* ; } 
nosenose-egg(){       package-fn $FUNCNAME $* ; }
nosenose-get(){       package-fn $FUNCNAME $* ; }

nosenose-install(){   package-fn $FUNCNAME $* ; }
nosenose-uninstall(){ package-fn $FUNCNAME $* ; }
nosenose-reinstall(){ package-fn $FUNCNAME $* ; }
nosenose-enable(){    package-fn $FUNCNAME $* ; }

nosenose-status(){    package-fn $FUNCNAME $* ; }
nosenose-auto(){      package-fn $FUNCNAME $* ; }
nosenose-diff(){      package-fn $FUNCNAME $* ; } 
nosenose-rev(){       package-fn $FUNCNAME $* ; } 
nosenose-cd(){        package-fn $FUNCNAME $* ; }

nosenose-fullname(){  package-fn $FUNCNAME $* ; }


