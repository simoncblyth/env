# === func-gen- : tools/otool fgp tools/otool.bash fgn otool fgh tools
otool-src(){      echo tools/otool.bash ; }
otool-source(){   echo ${BASH_SOURCE:-$(env-home)/$(otool-src)} ; }
otool-vi(){       vi $(otool-source) ; }
otool-env(){      elocal- ; }
otool-usage(){ cat << EOU

OTOOL
=======

* http://www.kitware.com/blog/home/post/510


BACKGROUND
------------

Extracts from *man otool*

*otool -D*
     Display just install name of a shared library.

*otool -L*
     Displays the names and version numbers of the shared libraries that the object file uses.  
     As well as the shared library ID if the file is a shared library.

*otool -l*
     Display the load commands.


FUNCTIONS
---------

*otool-install-name lib*
     install name of a lib from *otool -D*

*otool-install-name-deps lib*
     install names of dependants of a lib from *otool -L* 

*otool-rpath bin*
     greps the load commands from *otool -l* for LC_RPATH



EOU
}
otool-dir(){ echo $(local-base)/env/tools/tools-otool ; }
otool-cd(){  cd $(otool-dir); }
otool-mate(){ mate $(otool-dir) ; }
otool-get(){
   local dir=$(dirname $(otool-dir)) &&  mkdir -p $dir && cd $dir

}

otool-install-name(){       otool -D $1 ;}
otool-install-name-deps(){  otool -L $1 ;}
otool-rpath(){              otool -l $1 | grep LC_RPATH -A2 ; }  # -A/--after-context



otool-info(){
  echo otool -D 
  otool -D $1
  echo otool -L
  otool -L $1
  echo otool -l $1 \| grep LC_RPATH -A2
  otool -l $1 | grep LC_RPATH -A2
}
