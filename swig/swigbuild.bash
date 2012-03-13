# === func-gen- : swig/swigbuild fgp swig/swigbuild.bash fgn swigbuild fgh swig
swigbuild-src(){      echo swig/swigbuild.bash ; }
swigbuild-source(){   echo ${BASH_SOURCE:-$(env-home)/$(swigbuild-src)} ; }
swigbuild-vi(){       vi $(swigbuild-source) ; }
swigbuild-env(){      
   elocal- 
   swig-
}
swigbuild-usage(){
   cat << EOU

      http://www.swig.org/
            SWIG is a software development tool that connects programs written 
            in C and C++ with a variety of high-level programming languages.

     follow instructions from :
          $(svnbuild-dir)/subversion/bindings/swig/INSTALL
     specifically suitable versions of swig for binding with subversion 1.4.2 are  1.3.24 - 1.3.29
     
      (system version on cms01 is 1.3.21 ... so no go, and have to be cautions that it is not used by accident )
   
   
     Must be built after python and before subversion 

     mixed messages re versions http://trac.edgewall.org/wiki/TracSubversion
     trying : 
                SWIG_NAME : $SWIG_NAME
                SWIG_HOME : $SWIG_HOME

     swigbuild-get
     swigbuild-configure
     swigbuild-install
     swigbuild-check
     
        should report:
             SWIG Version 1.3.29
             Compiled with g++ [i686-pc-linux-gnu] 
             Please see http://www.swig.org for reporting bugs and further information


     $(type swigbuild-again)
EOU
}
swigbuild-dir-new(){ echo $(local-base)/env/swig/swig-swigbuild ; }
swigbuild-dir(){
   echo $SYSTEM_BASE/swig/build/$SWIG_NAME
}



swigbuild-cd(){  cd $(swigbuild-dir); }
swigbuild-mate(){ mate $(swigbuild-dir) ; }
swigbuild-get-new(){
   local dir=$(dirname $(swigbuild-dir)) &&  mkdir -p $dir && cd $dir

}

swigbuild-get(){
 
    local nam=$SWIG_NAME
    local tgz=$nam.tar.gz
    local url=http://jaist.dl.sourceforge.net/sourceforge/swig/$tgz

    cd $SYSTEM_BASE
    mkdir -p swig
    cd swig

    [ ! -f $tgz ] && curl -O $url
    mkdir -p  build
    [ ! -d build/$nam ] && tar -C build -zxvf $tgz 
}


swigbuild-configure(){

   python- 

   cd $(swigbuild-dir)
   ./configure -h
   ./configure --prefix=$SWIG_HOME --with-python=$PYTHON_HOME/bin/python --without-perl5 --without-gcj --without-java

}

swigbuild-install(){
   
   cd $(swigbuild-dir)
   make 
   make install

}


swigbuild-wipe(){
   local iwd=$PWD
   local dir=$SYSTEM_BASE/swig
   [ ! -d $dir ] && return 0
   cd $dir
   [ -d build ] && rm -rf build
   cd $iwd
}

swigbuild-wipe-install(){
   local iwd=$PWD
   local dir=$SYSTEM_BASE/swig
   [ ! -d $dir ] && return 0
   cd $dir
   [ "${SWIG_NAME:0:4}" != "swig" ] && echo bad name $SWIG_NAME cannot proceed && return 1
   [ -d $SWIG_NAME ] && rm -rf $SWIG_NAME
   cd $iwd
}


swigbuild-again(){

  swigbuild-wipe
  swigbuild-wipe-install
  
  swigbuild-get
  swigbuild-configure
  swigbuild-install

  swigbuild-check
}



swigbuild-check(){
     $SWIG_HOME/bin/swig -version
}




