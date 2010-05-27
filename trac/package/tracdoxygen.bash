# === func-gen- : trac/package/tracdoxygen fgp trac/package/tracdoxygen.bash fgn tracdoxygen fgh trac/package
tracdoxygen-src(){      echo trac/package/tracdoxygen.bash ; }
tracdoxygen-source(){   echo ${BASH_SOURCE:-$(env-home)/$(tracdoxygen-src)} ; }
tracdoxygen-vi(){       vi $(tracdoxygen-source) ; }
tracdoxygen-usage(){
  cat << EOU
     tracdoxygen-src : $(tracdoxygen-src)
     tracdoxygen-dir : $(tracdoxygen-dir)

     http://trac-hacks.org/wiki/DoxygenPlugin

     http://www.stack.nl/~dimitri/doxygen/


EOU
}

tracdoxygen-env(){
  elocal-
  package-
  export TRACDOXYGEN_BRANCH=0.11
}
tracdoxygen-revision(){  echo 7966 ; }
tracdoxygen-url(){       echo http://trac-hacks.org/svn/doxygenplugin/$(tracdoxygen-branch) ; }
tracdoxygen-package(){   echo doxygentrac ; }
tracdoxygen-fix(){ 
   echo -n
}

tracdoxygen-prepare(){
   tracdoxygen-enable $*

}



tracdoxygen-makepatch(){  package-fn $FUNCNAME $* ; }
tracdoxygen-applypatch(){ package-fn $FUNCNAME $* ; }

tracdoxygen-branch(){    package-fn $FUNCNAME $* ; }
tracdoxygen-basename(){  package-fn $FUNCNAME $* ; }
tracdoxygen-dir(){       package-fn $FUNCNAME $* ; }
tracdoxygen-egg(){       package-fn $FUNCNAME $* ; }
tracdoxygen-get(){       package-fn $FUNCNAME $* ; }

tracdoxygen-install(){   package-fn $FUNCNAME $* ; }
tracdoxygen-uninstall(){ package-fn $FUNCNAME $* ; }
tracdoxygen-reinstall(){ package-fn $FUNCNAME $* ; }
tracdoxygen-enable(){    package-fn $FUNCNAME $* ; }

tracdoxygen-status(){    package-fn $FUNCNAME $* ; }
tracdoxygen-auto(){      package-fn $FUNCNAME $* ; }
tracdoxygen-diff(){      package-fn $FUNCNAME $* ; }
tracdoxygen-rev(){       package-fn $FUNCNAME $* ; }
tracdoxygen-cd(){        package-fn $FUNCNAME $* ; }

tracdoxygen-fullname(){  package-fn $FUNCNAME $* ; }
tracdoxygen-update(){    package-fn $FUNCNAME $* ; }





