
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

#docutils-egg(){
#  ## override the egg as the setup.py is not handled 
#  echo  docutils-0.5-py2.5.egg
#}


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

docutils-get-prior(){

   local nik=docutils
   local nam=$nik-snapshot
   local tgz=$nam.tgz
   local url=ftp://ftp.berlios.de/pub/docutils/$tgz
   local dir=$LOCAL_BASE/python/$nik
   local uir=$dir/$nik


   local iwd=$(pwd)
   
   mkdir -p $dir 
   cd $dir 
   test -f $tgz || curl -o $tgz $url 
   test -d $nik || tar zxvf $tgz

   cd $uir
   #cd $iwd


}



