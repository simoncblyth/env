#
#
#
#   http://www.swig.org/
#      SWIG is a software development tool that connects programs written in C and C++ with a variety of high-level programming languages.
#
#   follow instructions from $SVN_HOME/subversion/bindings/swig/INSTALL  esp wrt versions 
#
#  mixed messages re versions:
#    http://trac.edgewall.org/wiki/TracSubversion
#  try 
#       1.3.25
#
#
#
#     swig-x
#     swig-i
#
#     swig-get
#     swig-configure
#     swig-install
#     swig-check
#
#
#


swig-env(){
   local SWIG_NAME=swig-1.3.29
   export SWIG_HOME=$LOCAL_BASE/swig/$SWIG_NAME
}

swig-get(){
 
    nam=$SWIG_NAME
	nik=swig
    tgz=$nam.tar.gz
    url=http://jaist.dl.sourceforge.net/sourceforge/swig/$tgz

    cd $LOCAL_BASE
    test -d $nik || ( $SUDO mkdir $nik && $SUDO chown $USER $nik  )
    cd $nik

    test -f $tgz || curl -o $tgz $url
    test -d build || mkdir build
    test -d build/$nam || tar -C build -zxvf $tgz 
}

swig-configure(){

   cd $LOCAL_BASE/swig/build/$SWIG_NAME
   ./configure -h
   ./configure --prefix=$SWIG_HOME --with-python=$PYTHON_HOME/bin/python --without-perl5 --without-gcj --without-java
}

swig-install(){
   cd $LOCAL_BASE/swig/build/$SWIG_NAME
   make 
   make install
}


swig-check(){
  $SWIG_HOME/bin/swig -version
}

# reports:
#
# SWIG Version 1.3.29
#
# Compiled with g++ [i686-pc-linux-gnu]
# Please see http://www.swig.org for reporting bugs and further information

