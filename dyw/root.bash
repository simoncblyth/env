droot-vi(){ vi $BASH_SOURCE ; }
droot-info(){

  cat << EOU

   droot-mode     : $(droot-mode $*)
      if not "binary" source is assumed

   droot-version  : $(droot-version $*)
   droot-name     : $(droot-name $*)
   droot-nametag  : $(droot-nametag $*)
   droot-url      : $(droot-url $*)
   droot-rootsys  : $(droot-rootsys $*)
   droot-base     : $(droot-base $*) 
 

    ROOTSYS    : $ROOTSYS
    which root : $(which root)

    After changing the root version you will need to run :
        cmt-gensitereq

    This informs CMT of the change via re-generation of
     the non-managed :
         $ENV_HOME/externals/site/cmt/requirements
    containing the ROOT_prefix variable 

    This works via the ROOT_CMT envvar that is set by droot-env, such as: 
       env | grep _CMT
       ROOT_CMT=ROOT_prefix:/data/env/local/root/root_v5.21.04.source/root

   Changing root version will require rebuilding libs that are 
   linked against the old root version, that includes dynamically created libs


   droot-           :  hook into these functions invoking droot-env
   droot-get        :  download and unpack
   droot-configure  :     
   droot-build      :


   droot-path       :
         invoked by the precursor, sets up PATH (DY)LD_LIBRARY_PATH and PYTHONPATH

   droot-pycheck
         http://root.cern.ch/root/HowtoPyROOT.html

   droot-evetest

   droot-ps         : list root.exe processes
   droot-killall    : kill root.exe processes



EOU

}

droot-usage(){
   droot-info
}


droot-ps(){
  ps aux | grep root.exe
}

droot-killall(){
   killall root.exe
}

droot-archtag(){

   [ "$(droot-mode)" != "binary" ] && echo "source" && return 0 

   case ${1:-$NODE_TAG} in 
      C) echo Linux-slc4-gcc3.4 ;;
      G) echo macosx-powerpc-gcc-4.0 ;;
      *) echo Linux-slc4-gcc3.4 ;;
   esac
}

droot-nametag(){ 
   echo $(droot-name $*).$(droot-archtag $*) 
}


droot-get(){

   local msg="=== $FUNCNAME :"
   
   local base=$(droot-base)
   [ ! -d "$base" ] && mkdir -p $base 
   cd $base  ## 2 levels above ROOTSYS , the 
   local n=$(droot-nametag)
   [ ! -f $n.tar.gz ] && curl -O $(droot-url)
   [ ! -d $n/root   ] && mkdir $n && tar  -C $n -zxvf $n.tar.gz 
 
   ## unpacked tarballs create folder called "root"
}

droot-version(){
  local def="5.21.04"
  local jmy="5.22.00"   ## has eve X11 issues 
  local new="5.23.02" 
  case ${1:-$NODE_TAG} in 
     C) echo $def ;;
     *) echo $def ;;
  esac
}

#droot-mode(){    echo binary  ; }
droot-mode(){    echo -n  ; }
droot-name(){    echo root_v$(droot-version $*) ; }
droot-base(){    echo $(dirname $(dirname $(droot-rootsys $*))) ; }
droot-rootsys(){ echo $(local-base $1)/root/$(droot-nametag $1)/root ; }
droot-url(){     echo ftp://root.cern.ch/root/$(droot-nametag $*).tar.gz ; }

droot-cd(){ cd $(droot-rootsys)/tutorials/eve ; }








droot-env(){

  elocal-

  alias root="root -l"
  alias rh="tail -100 $HOME/.root_hist"
 
  export ROOT_NAME=$(droot-name)
  export ROOTSYS=$(droot-rootsys)
  
  ## pre-nuwa ... to be dropped      	
  export ROOT_CMT="ROOT_prefix:$ROOTSYS"

  droot-path
}



droot-path(){

  [ ! -d $ROOTSYS ] && return 0
  
  env-prepend $ROOTSYS/bin
  env-llp-prepend $ROOTSYS/lib
  env-pp-prepend $ROOTSYS/lib
}











droot-pycheck(){
  python -c "import ROOT as _ ; print _.__file__ "

}






## test -f $tgz || scp S:/usr/local/root/$tgz .   
## direct way doesnt work on pal... "curl: (19) Security: Bad IP connecting."







droot-tutetest(){
  local dir=$1
  local name=$2
  local msg="=== $FUNCNAME :"
  local iwd=$PWD

  cd $dir  
  [ ! -f "$name" ] && echo $msg no such script $PWD/$name  && cd $iwd && return 1
 
  droot-config --version
 
  local cmd="root $name"
  echo $msg $cmd
  eval $cmd
}

droot-evetest(){ droot-tutetest $ROOTSYS/tutorials/eve $* ; }
droot-gltest(){  droot-tutetest $ROOTSYS/tutorials/gl  $* ; }






droot-mysql(){

  ## https://wiki.bnl.gov/dayabay/index.php?title=Database
  
  cd $ROOTSYS
  local cmd="./configure $(cat config.status) --enable-mysql --with-mysql-incdir=$MYSQL_HOME/include --with-mysql-libdir=$MYSQL_HOME/lib "
  
  echo $cmd
  eval $cmd

  make all-mysql


}


droot-configure(){

  ## an environment controlled config
  cd $ROOTSYS 
  ./configure --help


  if [ "$LOCAL_NODE" == "g4pb" ]; then

  ./configure --enable-gdml \
	          --enable-opengl \
		      --enable-python \
			  --enable-roofit \
			  --enable-ruby \
			  --enable-xml \
			  --disable-g4root 
			  
   elif [ "$LOCAL_NODE" == "pal" ]; then

 ./configure --enable-gdml \
	          --enable-opengl \
		      --enable-python \
			  --enable-roofit \
			  --enable-ruby \
			  --enable-xml \
			  --disable-g4root \
			  --disable-mysql \
			  --disable-castor
   
  else 

 ./configure --enable-gdml \
	          --enable-opengl \
		      --enable-python \
			  --enable-roofit \
			  --enable-ruby \
			  --enable-xml \
			  --disable-g4root \
			  --disable-mysql \
			  --disable-ldap \
			  --disable-globus \
			  --disable-castor
 
   
   fi



  ## need to disable g4root as presumably the configure switches it on
  ## when it sees the G4 environmnent variables
}


droot-build(){
  cd $ROOTSYS
  make
  make install
}


# [pal] /usr/local/root/root/bin > ./root -b
# *******************************************
# *                                         *
# *        W E L C O M E  to  R O O T       *
# *                                         *
# *   Version  5.14/00b   17 January 2007   *
# *                                         *
# *  You are welcome to visit our Web site  *
# *          http://root.cern.ch            *
# *                                         *
# *******************************************
# 
# Compiled on 1 February 2007 for linuxx8664gcc with thread
# support.
# 
# CINT/ROOT C/C++ Interpreter version 5.16.16, November 24, 2006
# 







####################################################################################

configure-droot-failure2(){

  ## an environment controlled config
  cd $ROOTSYS 
  ./configure --help
  ./configure --enable-gdml \
	          --enable-opengl \
		      --enable-python \
			  --enable-roofit \
			  --enable-ruby \
			  --enable-xml \
			                \
              --enable-g4root \
			  --with-g4-incdir=$GQ_HOME/include \
			  --with-g4-libdir=$GQ_HOME/lib/$G4SYSTEM \

#bin/rmkdepend -R -fg4root/src/TG4RootDetectorConstruction.d -Y -w 1000 --
#-pipe -W -Wall -Wno-long-double -Woverloaded-virtual -fsigned-char -fno-common
#-Iinclude    -I/usr/local/g4/dist/1/geant4.8.1.p01/include
#-I/usr/local/clhep/clhep-1.9.2.3-osx104_ppc_gcc401/include -D__cplusplus --
#g4root/src/TG4RootDetectorConstruction.cxx
#g++ -O2 -pipe -W -Wall -Wno-long-double -Woverloaded-virtual -fsigned-char
#-fno-common -Iinclude    -I/usr/local/g4/dist/1/geant4.8.1.p01/include
#-I/usr/local/clhep/clhep-1.9.2.3-osx104_ppc_gcc401/include -DUSEPCH -include
#precompile.h -o g4root/src/TG4RootDetectorConstruction.o -c
#g4root/src/TG4RootDetectorConstruction.cxx
#/usr/local/g4/dist/1/geant4.8.1.p01/include/G4VPVParameterisation.hh:73:
#warning: 'class G4VPVParameterisation' has virtual functions but non-virtual
#destructor
#include/TG4RootNavigator.h: In member function 'G4NavigationHistory*
#TG4RootNavigator::GetHistory()':
#/usr/local/g4/dist/1/geant4.8.1.p01/include/G4Navigator.hh:331: error:
#'G4NavigationHistory G4Navigator::fHistory' is private
#include/TG4RootNavigator.h:54: error: within this context
#make: *** [g4root/src/TG4RootDetectorConstruction.o] Error 1
#[g4pb:/usr/local/root514/root] blyth$ 
#
#
#   it seems that VGM supplies the functionality of g4root ... so try without
#   g4root
#


}




configure-droot-failure1(){

  export ROOT_FOLDER=/usr/local/root
  export ROOT_VERSION=root_v5.14.00a
  export ROOT_SYS=$ROOT_FOLDER/$ROOT_VERSION/root
  ##export ROOTSYS=$ROOT_SYS
#  when using static root, ie configured with --prefix=/usr/local/root
#  should not set the ROOTSYS env variable

  cd $ROOT_SYS 
  ./configure --help
  ./configure --prefix=/usr/local/root \
              --enable-gdml \
	          --enable-opengl \
		      --enable-python \
			  --enable-roofit \
			  --enable-ruby \
			  --enable-xml \
			                \
              --enable-g4root \
			  --with-g4-incdir=$GQ_HOME/include \
			  --with-g4-libdir=$GQ_HOME/lib/$G4SYSTEM \
     

#
#  fails with :
# g++ -O2 -dynamiclib -single_module -undefined dynamic_lookup -install_name
# /usr/local/root/lib/root/libGraf3d.dylib -o lib/libGraf3d.dylib
# g3d/src/TAxis3D.o g3d/src/TBRIK.o g3d/src/TCONE.o g3d/src/TCONS.o
# g3d/src/TCTUB.o g3d/src/TELTU.o g3d/src/TGTRA.o g3d/src/TGeometry.o
# g3d/src/THYPE.o g3d/src/THelix.o g3d/src/TMarker3DBox.o g3d/src/TMaterial.o
# g3d/src/TMixture.o g3d/src/TNode.o g3d/src/TNodeDiv.o g3d/src/TPARA.o
# g3d/src/TPCON.o g3d/src/TPGON.o g3d/src/TPointSet3D.o g3d/src/TPoints3DABC.o
# g3d/src/TPolyLine3D.o g3d/src/TPolyMarker3D.o g3d/src/TRotMatrix.o
# g3d/src/TSPHE.o g3d/src/TShape.o g3d/src/TTRAP.o g3d/src/TTRD1.o
# g3d/src/TTRD2.o g3d/src/TTUBE.o g3d/src/TTUBS.o g3d/src/TUtil3D.o
# g3d/src/TXTRU.o g3d/src/X3DBuffer.o g3d/src/G__G3D.o -ldl -Llib -lGraf -lHist
# -Llib -lCore -lCint
# ld: warning can't open dynamic library:
# /usr/local/root/lib/root/libMatrix.dylib referenced from: lib/libGraf.dylib
# (checking for undefined symbols may be affected) (No such file or directory,
# errno = 2)
# ld: Undefined symbols:
# __ZN8TVectorTIdE5ClearEPKc referenced from libHist expected to be defined in /usr/local/root/lib/root/libMatrix.dylib
# __ZN8TVectorTIdE5ClearEPKc.eh referenced from libHist expected to be defined in /usr/local/root/lib/root/libMatrix.dylib
# __ZN8TVectorTIdED0Ev referenced from libHist expected to be defined in /usr/local/root/lib/root/libMatrix.dylib
#
#
#   [g4pb:/usr/local/root/root_v5.14.00a/root] blyth$ otool -L lib/libMatrix.dylib 
#    lib/libMatrix.dylib:
#   /usr/local/root/lib/root/libMatrix.dylib (compatibility version 0.0.0, current version 0.0.0)
#	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 88.1.7)
#   /usr/local/root/lib/root/libCore.dylib (compatibility version 0.0.0, current version 0.0.0)
#	/usr/local/root/lib/root/libCint.dylib (compatibility version 0.0.0, current version 0.0.0)
#   /usr/lib/libstdc++.6.dylib (compatibility version 7.0.0, current version 7.4.0)
#   /usr/lib/libgcc_s.1.dylib (compatibility version 1.0.0, current version 1.0.0)
# 
#
# libdir (<prefix>/lib/root)
# Library installation directory. All the class libraries of ROOT will be
# installed into this directory. You should make your dynamic linker aware of
# this directory. On Linux - and some other Un*ces - this directory can be added
# to /etc/ld.so.conf and ldconfig should be run afterward (you need to be root -
# the user - to do this). Please note, that this directory should probably not
# be something like /usr/lib or /usr/local/lib, since you'll most likely get a
# name clash with ROOT libraries and other libraries (e.g. libMatrix.so); rather
# use something like /usr/local/lib/root
#
#
#  possibly a name clash issue

}


