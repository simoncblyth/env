
docutils-usage(){

   package-fn $FUNCNAME $*

cat << EOU

   http://docutils.sourceforge.net/

  another provider of syntax highlighting to trac...
  installed principally to avoid the error messages in the trac.log and
  associated performance hit

      http://trac.edgewall.org/wiki/TracSyntaxColoring
       
EOU

}




docutils-env(){
   elocal-
   package-
   
   export DOCUTILS_BRANCH=docutils-snapshot.tgz
}

docutils-url(){
   echo ftp://ftp.berlios.de/pub/docutils/$(docutils-branch)
}

docutils-basename(){
  ## override the basename as docutils-snapshot.tgz expands to a docutils dir  
   echo docutils
}




docutils-branch(){    package-fn $FUNCNAME $* ; }

docutils-dir(){       package-fn $FUNCNAME $* ; } 

docutils-get(){       package-fn $FUNCNAME $* ; }

docutils-install(){   package-fn $FUNCNAME $* ; }
docutils-uninstall(){ package-fn $FUNCNAME $* ; }
docutils-reinstall(){ package-fn $FUNCNAME $* ; }
docutils-enable(){    package-fn $FUNCNAME $* ; }

docutils-status(){    package-fn $FUNCNAME $* ; }
docutils-auto(){      package-fn $FUNCNAME $* ; }
docutils-diff(){      package-fn $FUNCNAME $* ; } 
docutils-rev(){       package-fn $FUNCNAME $* ; } 
docutils-cd(){        package-fn $FUNCNAME $* ; }

docutils-fullname(){  package-fn $FUNCNAME $* ; }
docutils-update(){    package-fn $FUNCNAME $* ; }




