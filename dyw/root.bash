alias p-root="scp $HOME/.bash_root P:"

#   usage:
#            root-get
#            root-configure
#            root-build
#
#
# Qt BNL and Qt GSI ????  (both are currently for Qt 3.xx , not 4.x that I have)
#            --with-qtgsi \
#  the BNL one looks more mature ??? 
# http://www-linux.gsi.de/~go4/qtroot/html/qtroot.html 
#


alias root="root -l"
alias rh="tail -100 $HOME/.root_hist"

export ROOT_FOLDER=$LOCAL_BASE/root
export ROOT_NAME=root_v5.14.00b

if [ "$LOCAL_NODE" == "pal" ]; then
	     ## prior to extreme versioning 
  export ROOTSYS=$ROOT_FOLDER/root
else
  export ROOTSYS=$ROOT_FOLDER/$ROOT_NAME/root
fi
	
if [ "$CMTCONFIG" == "Darwin" ]; then
  export DYLD_LIBRARY_PATH=$ROOTSYS/lib:$DYLD_LIBRARY_PATH
else
  export   LD_LIBRARY_PATH=$ROOTSYS/lib:$LD_LIBRARY_PATH
fi


export ENV2GUI_VARLIST="ROOTSYS:$ENV2GUI_VARLIST"
export PATH=$ROOTSYS/bin:$PATH
export ROOT_CMT="ROOT_prefix:$ROOTSYS"

[ "$DYW_DBG" == "1" ] && echo $DYW_BASE/root.bash
root-get(){

  n=$ROOT_NAME
  cd $LOCAL_BASE
  test -d root || ( $SUDO mkdir root && $SUDO chown $USER root )
  cd root

  tgz=$n.source.tar.gz

  if [ "$LOCAL_NODE" == "pal" ]; then
     test -f $tgz || scp S:/usr/local/root/$tgz .
  
     ## direct way doesnt work on pal... "curl: (19) Security: Bad IP connecting."
     ##  IS PAL A BLACKLISTED MACHINE ? OR NO DNS RECORD ?

  else
     test -f $tgz || curl -o $tgz ftp://root.cern.ch/root/$tgz
  fi
 
  ## using prior knowledge that the unpacked creates a folder called "root"
  test -d $n/root || ( mkdir $n && tar -C $n -zxvf $tgz  )


}



root-mysql(){

  ## https://wiki.bnl.gov/dayabay/index.php?title=Database
  
  cd $ROOTSYS
  local cmd="./configure $(cat config.status) --enable-mysql --with-mysql-incdir=$MYSQL_HOME/include --with-mysql-libdir=$MYSQL_HOME/lib "
  
  echo $cmd
  eval $cmd

  make all-mysql


}


root-configure(){

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


root-build(){
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

configure-root-failure2(){

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




configure-root-failure1(){

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


