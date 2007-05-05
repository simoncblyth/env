alias p-openmotif="scp $HOME/.bash_openmotif P:"

###### G4 override , as darwinports motif incompatible with Coin3d SoXt

XM_NAME=openMotif-2.2.3

if [ "$LOCAL_NODE" == "grid1" ]; then
   ##export XMHOME="$LOCAL_BASE/openmotif/$XM_NAME" 
   export XMHOME="/usr/X11R6" 
else
   export XMHOME="$LOCAL_BASE/openmotif" 
fi

export OPENMOTIF_HOME=$XMHOME


export XMFLAGS="-I${XMHOME}/include"  
export XMLIBS="-L${XMHOME}/lib -lXm -lXpm"    ## libXpm is not there 
export ENV2GUI_VARLIST="XMHOME:$ENV2GUI_VARLIST"


if [ "$CMTCONFIG" == "amd64_linux26" ]; then
   export X11LIBS="-L/usr/X11R6/lib64  -lXmu -lXt -lXext -lX11 -lSM -lICE "
else
   export X11LIBS="-L/usr/X11R6/lib  -lXmu -lXt -lXext -lX11 -lSM -lICE "
fi




#######export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/usr/X11R6/lib
## having X11R6 in this path causes warnings:
##   dyld: warning DYLD_ setting caused circular dependency in /usr/X11R6/lib/libGL.dylib


if [ "$CMTCONFIG" == "Darwin" ]; then
   export DYLD_LIBRARY_PATH=${XMHOME}/lib:${DYLD_LIBRARY_PATH}
else
   export   LD_LIBRARY_PATH=${XMHOME}/lib:${LD_LIBRARY_PATH}
fi



openmotif-get(){

  n=$XM_NAME
  cd $LOCAL_BASE
  test -d openmotif || ( $SUDO mkdir openmotif && $SUDO chown $USER openmotif )
  test -d graphics  || ( $SUDO mkdir graphics  && $SUDO chown $USER graphics )
  cd graphics 

  #scp S:$LOCAL_BASE/graphics/$n.tar.gz  .
  
  
  tar zxvf $n

}

openmotif-configure(){

  n=$XM_NAME
  cd $LOCAL_BASE/graphics/$n
  ./configure --prefix=$LOCAL_BASE/openmotif
  
  ## next time, retain versioning here
}

openmotif-build(){
  n=$XM_NAME
#  fails with LANG as en_US.UTF-8 ... try en_US, in .bash_profile
# ... as make tends to make lots of new shells  
  cd $LOCAL_BASE/graphics/$n
  make
}


openmotif-install(){
  n=$XM_NAME
#  fails with LANG as en_US.UTF-8 ... try en_US, in .bash_profile
# ... as make tends to make lots of new shells  
  cd $LOCAL_BASE/graphics/$n
  make install
}




#
#  
#
#mkdir .libs
#gcc -g -O2 -Wall -Wno-unused -Wno-comment -o .libs/xmanimate xmanimate.o
#../../lib/Xmd/libXmd.a ../../../lib/Mrm/.libs/libMrm.so
#/usr/local/graphics/openMotif-2.2.3/lib/Xm/.libs/libXm.so -L/usr/X11R6/lib64
#../../../lib/Xm/.libs/libXm.so -lXmu -lXt -lSM -lICE -lXext -lXp -lX11
#-Wl,--rpath -Wl,/usr/local/openmotif/lib
#creating xmanimate
#../../../clients/uil/uil -o dog.uid dog.uil -I./../../../clients/uil
#-I../../../clients/uil
#
#Error: $LANG contains an unknown character set
#
#Info: no UID file was produced
#
#Info: errors: 1  warnings: 0  informatio
#



xmenv(){
	env | grep XM
}


