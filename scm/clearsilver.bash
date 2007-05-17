#
#
#  clearsilver-get
#  clearsilver-wipe 
#  clearsilver-configure
#  clearsilver-install
#
#
#   on darwin had to do manual editing of 
#      configure 
#      configure.in
# following the 0.10.2 instructions from :
#   http://trac.edgewall.org/wiki/ClearSilver
# 
#  also removed a "ruby" from the rules.mk ... it seems configure options 
#   are not honoured ...
#
#  basically clearsilver build system is a mess ... thank god that Trac are
#  abandoning clearsilver for the next release
#
#
#   clearsilver-0.10.2 get safari "lost network connection" and  error in apache2 error log on hitting trac...
#  ================================
#
# Fatal Python error: Interpreter not initialized (version mismatch?)
# [Tue May 15 16:13:36 2007] [notice] child pid 1196 exit signal Abort trap (6)
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/trac/web/clearsilver.py:128: RuntimeWarning: Python C API version mismatch for module neo_util: This Python has API version 1013, module neo_util has version 1012.
#
#
#     clearsilver-0.10.4 get
#  ============================ 
#  make[1]: *** No rule to make target `features.h', needed by `cgiwrap.o'.
#
#       THIS LOOKS A BLOCKER... 
#
#
#
#     clearsilver-0.10.3 get
#  ============================ 
#
#  gcc -shared -fPIC -o static.cso static.o -L../libs/ -lneo_cgi -lneo_cs  -lneo_utl  -lz
# powerpc-apple-darwin8-gcc-4.0.1: unrecognized option '-shared'
# 
# [g4pb:/usr/local/clearsilver/build/clearsilver-0.10.3/python] blyth$ gcc
# -shared -fPIC build/temp.macosx-10.4-ppc-2.5/neo_cgi.o
# build/temp.macosx-10.4-ppc-2.5/neo_cs.o
# build/temp.macosx-10.4-ppc-2.5/neo_util.o -L$NEOTONIC_ROOT/libs/ -L../libs -lz
# -lneo_cgi -lneo_cs -lneo_utl -o build/lib.macosx-10.4-ppc-2.5/neo_cgi.so
# powerpc-apple-darwin8-gcc-4.0.1: unrecognized option '-shared'
# /usr/bin/ld: warning -L: directory name (/libs/) does not exist
# /usr/bin/ld: Undefined symbols:
# _main
# _PyArg_ParseTuple
# _PyArg_ParseTupleAndKeywords
# _PyCObject_FromVoidPtr
# _PyDict_SetItemString
# _PyErr_Clear
# 
#     hmm tis not linking with the python libraries...
#
#
#  > > The best way is to use distutils (or setuptools). Create a setup.py
#  > > that compiles it for you. This works on every platform, even windows.
#
# 
#    0.10.3 .... duplicate the issue 
#   ==================================
#
#
#   [g4pb:/usr/local/clearsilver/build/clearsilver-0.10.3/python] blyth$
# CC="gcc" LDSHARED="gcc -shared -fPIC" /usr/local/python/Python-2.5.1/bin/python setup.py build_ext --inplace
#   adding lib_path $(LIB_DIR)
#   adding lib z
#   running build_ext
#   building 'neo_cgi' extension
#   gcc -shared -fPIC build/temp.macosx-10.4-ppc-2.5/neo_cgi.o
#   build/temp.macosx-10.4-ppc-2.5/neo_cs.o
#   build/temp.macosx-10.4-ppc-2.5/neo_util.o -L $(NEOTONIC_ROOT)/libs/ -L../libs -lz -lneo_cgi -lneo_cs -lneo_utl -o
#   build/lib.macosx-10.4-ppc-2.5/neo_cgi.so
#   powerpc-apple-darwin8-gcc-4.0.1: unrecognized option '-shared'
#   /usr/bin/ld: warning -L: directory name ( $(NEOTONIC_ROOT)/libs/) does not
#   exist
#   /usr/bin/ld: Undefined symbols:
#   _main
#   _PyArg_ParseTuple
#   _PyArg_ParseTupleAndKeywords
#   _PyCObject_FromVoidPtr
#   _PyDict_SetItemString
#
#
#
#   remove the LDSHARED ... seems to build OK ???
#
#    CC="gcc"  /usr/local/python/Python-2.5.1/bin/python setup.py build_ext --inplace
#
#   [g4pb:/usr/local/clearsilver/build/clearsilver-0.10.3/python] blyth$ make LDSHARED=
#	make: Nothing to be done for `everything'.
#
#
#   0.10.3 succeeded to build by manual intervention....
#
#       [g4pb:/usr/local/clearsilver/build/clearsilver-0.10.3/python] blyth$ # make LDSHARED=
#  
#
#      make LDSHARED="gcc -dynamiclib -undefined dynamic_lookup "
#
#
#    clearsilver-wipe
#    clearsilver-get
#    clearsilver-configure
#    clearsilver-install
#

#export MACOSX_DEPLOYMENT_TARGET=10.3


if [ "$(uname)" == "Darwin" ]; then
 CLEARSILVER_NAME=clearsilver-0.10.4
#CLEARSILVER_NAME=clearsilver-0.10.2     ## got furthest with this... 
#CLEARSILVER_NAME=clearsilver-0.9.14
#CLEARSILVER_NAME=clearsilver-0.10.3

## from http://clearsilver.darwinports.com/  

# experimenting with python -c "import sys ; print sys.path "  
# indicates that PYTHON_PATH is ignored ... must be PYTHONPATH
#
PYTHONPATH=$(python -c "import sys ; print (sys.prefix or sys.exec_prefix) + '/lib/python' + sys.version[0:3]")

else
 CLEARSILVER_NAME=clearsilver-0.10.4
fi
#echo ====  set CLEARSILVER_NAME $CLEARSILVER_NAME 

CLEARSILVER_NIK=clearsilver
export CLEARSILVER_HOME=$LOCAL_BASE/$CLEARSILVER_NIK/$CLEARSILVER_NAME

clearsilver-cd(){
   nam=$CLEARSILVER_NAME
   nik=$CLEARSILVER_NIK
   cd $LOCAL_BASE/$nik/build/$nam
}

clearsilver(){

   NEOTONIC_ROOT=..

   clearsilver-wipe
   clearsilver-get
#   clearsilver-patch
   clearsilver-configure
   clearsilver-fix
   clearsilver-install


}




clearsilver-patch(){

   if [ "$CLEARSILVER_NAME" == "clearsilver-0.10.4" ]; then

     cd $LOCAL_BASE/$CLEARSILVER_NIK/build

## -p0 indicates the slash stipping on file paths in the patch ... 0 means no
## stripping
	 patch --verbose -p0 < macosx-patch-clearsilver-0.10.4-two.txt
 
   fi
#
#   the patch is malfomed ...
#
#
#   diff -Nru 
#     N : treat absent files as empty
#     r : recursive
#     u : provide 3 lines of context 


}

#
#   -fno-common
#   -undefined suppress
#

clearsilver-fix(){

#  setup.py parsed rules.mk ... making for painful configuration
#


   clearsilver-cd 

   #if [ "$CLEARSILVER_NAME" == "clearsilver-0.10.3" ]; then 
   
   perl -pi -e 's|^(CPPFLAGS.*)$|$1 -I/usr/local/python/Python-2.5.1/include/python2.5  -fno-common|' rules.mk
   perl -pi -e 's/^(LIBS.*)$/$1/' rules.mk
   perl -pi -e 's|^(LDFLAGS.*)$|$1 |' rules.mk
   #perl -pi -e 's|^(LDSHARED\s*=\s*)(.*)$|$1 gcc -dynamiclib -undefined dynamic_lookup  -L/usr/local/python/Python-2.5.1/lib/python2.5/config  -lpython2.5 |' rules.mk
   #perl -pi -e 's|^(LDSHARED\s*=\s*)(.*)$|$1 gcc -bundle  -L/usr/local/python/Python-2.5.1/lib/python2.5/config  -lpython2.5 |' rules.mk
   #perl -pi -e 's|^(LDSHARED\s*=\s*)(.*)$|$1 gcc -bundle -flat_namespace -undefined dynamic_lookup |' rules.mk
   perl -pi -e 's|^(LDSHARED\s*=\s*)(.*)$|$1 gcc -bundle -bundle_loader /usr/local/python/Python-2.5.1/bin/python  -flat_namespace -undefined warning |' rules.mk



   perl -pi -e 's|^LIBRARIES.*|LIBRARIES = ["neo_cgi", "neo_cs", "neo_utl", "python2.5","z" ]|' python/setup.py
   perl -pi -e 's|^LIB_DIRS.*|LIB_DIRS = ["../libs", "/usr/local/python/Python-2.5.1/lib/python2.5/config" ]|' python/setup.py
  
   #fi
   
   
   if [ "$CLEARSILVER_NAME" == "clearsilver-0.10.4" ]; then
      perl -pi -e 's|^(#include <features.h>)|//$1|' cgi/cgiwrap.c

      perl -pi -e 's/^(#include )<Python.h>/$1"Python.h"/' python/*.c
	  
   fi



}


clearsilver-test(){

 python -vdc "import neo_cgi"

 # dlopen("/usr/local/python/Python-2.5.1/lib/python2.5/site-packages/neo_cgi.so", 2);
 # Fatal Python error: Interpreter not initialized (version mismatch?)
 # Abort trap
 #
}


clearsilver-get(){
    
	nam=$CLEARSILVER_NAME
	nik=$CLEARSILVER_NIK
	tgz=$nam.tar.gz
	url=http://www.clearsilver.net/downloads/$tgz

    cd $LOCAL_BASE
    test -d $nik || ( $SUDO mkdir $nik && $SUDO chown $USER $nik )
    cd $nik 

    test -f $tgz || curl -o $tgz $url
    test -d build || mkdir build
    test -d build/$nam || tar -C build -zxvf $tgz 
}

clearsilver-wipe(){
   nam=$CLEARSILVER_NAME
   nik=$CLEARSILVER_NIK

   cd $LOCAL_BASE/$nik
   rm -rf build/$nam
}

clearsilver-configure(){
   
   clearsilver-cd
   ./configure --prefix=$CLEARSILVER_HOME  --with-python=$PYTHON_HOME/bin/python --enable-python --disable-ruby --disable-perl --disable-apache --disable-csharp --disable-java
}

# some pthread issue ... but seems OK in configure
# checking pthread.h usability... yes
# checking pthread.h presence... yes
# checking for pthread.h... yes


clearsilver-install(){

   clearsilver-cd
   make
   make install

   ls -alst $PYTHON_HOME/lib/python2.5/site-packages/neo_cgi.so
   ## this places a library at :  $PYTHON_HOME/lib/python2.5/site-packages/neo_cgi.so
}



clearsilver-darwin-fix(){
   ## fix braindead usage of script withy hardcoded python path in 	0.9.14
   clearsilver-cd
   perl -pi -e 's|(.*)(scripts/document.py)(.*)$|$1python $2$3|g && print ' Makefile
}







# gcc -g -O2 -Wall -I..  -fPIC -o neo_net.o -c neo_net.c
# gcc -g -O2 -Wall -I..  -fPIC -o neo_server.o -c neo_server.c
# ar cr ../libs/libneo_utl.a neo_err.o neo_files.o neo_misc.o neo_rand.o ulist.o
# neo_hdf.o neo_str.o neo_date.o wildmat.o neo_hash.o ulocks.o rcfs.o skiplist.o
# dict.o filter.o neo_net.o neo_server.o 
# ranlib ../libs/libneo_utl.a
# gcc -g -O2 -Wall -I..  -fPIC -o csparse.o -c csparse.c
# ar cr ../libs/libneo_cs.a csparse.o
# ranlib ../libs/libneo_cs.a
# gcc -g -O2 -Wall -I..  -fPIC -o cstest.o -c cstest.c
# gcc -o cstest cstest.o -L../libs/  -lz -lneo_cs -lneo_utl  # -lefence
# gcc -g -O2 -Wall -I..  -fPIC -o cs.o -c cs.c
# gcc -o cs cs.o -L../libs/  -lz -lneo_cs -lneo_utl  # -lefence
# Running cs regression tests
# Passed
# make[1]: *** No rule to make target `features.h', needed by `cgiwrap.o'.
# Stop.
# make[1]: *** No rule to make target `../libs/libneo_cgi.a', needed by
# `neo_cgi.so'.  Stop.
# make: *** [cs] Error 2
# make[1]: Nothing to be done for `everything'.
# make[1]: Nothing to be done for `everything'.
# make[1]: *** No rule to make target `features.h', needed by `cgiwrap.o'.
# Stop.
# make[1]: *** No rule to make target `../libs/libneo_cgi.a', needed by
# `neo_cgi.so'.  Stop.
# make: *** [cs] Error 2
# 
# 
# 
# 
# 
# 
# 
# 
# 
# /usr/bin/install -c -m 644 ../libs/libneo_cgi.a
# /usr/local/clearsilver/clearsilver-0.9.14/lib
# /usr/bin/install -c static.cgi /usr/local/clearsilver/clearsilver-0.9.14/bin
# ../mkinstalldirs 
# /usr/bin/install -c neo_cgi.so 
# usage: install [-bCcpSsv] [-B suffix] [-f flags] [-g group] [-m mode]
#                [-o owner] file1 file2
# 			          install [-bCcpSsv] [-B suffix] [-f flags] [-g group] [-m
# 					  mode]
# 					                 [-o owner] file1 ... fileN directory
# 									        install -d [-v] [-g group] [-m
# 											mode] [-o owner] directory ...
# 											make[1]: *** [install] Error 64
# 											make: *** [install] Error 2
