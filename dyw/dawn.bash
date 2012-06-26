[ "$DYW_DBG" == "1" ] && echo $DYW_BASE/dawn.bash

export DAWN_NAME=dawn_3_88a
export DAWN_URLBASE=http://geant4.kek.jp/~tanaka/src
export DAWN_HOME_OLD=/usr/local/g4/dawn/dawn_3_88a
export DAWN_BUILD=/usr/local/dawn/build/$DAWN_NAME

export DAWN_HOME=/usr/local/dawn/bin
export PATH=$DAWN_HOME:$PATH

#alias dawn="$DAWN_HOME/dawn" 

## .DAWN_1.default
## .DAWN_1.history
  
#  from dawn -H
#setenv DAWN_PS_PREVIEWER  viewer_name
#setenv DAWN_BATCH  0/1/a (GUI/batch/batch+append) 
#setenv DAWN_DEVICE 1/2/3/4/5 (PS/X/PS2/X2/GL) 
#setenv DAWN_UP_DIRECTION  Y/Z 
#setenv DAWN_BFC_MODE 0/1 (Backface-culling mode off/on) 
#setenv DAWN_USE_STENCIL 0/1 (Skip/do drawing edges in OpenGL mode) 
						  
dawn-usage(){ cat << EOU

## cannot control the filename... g4_00.prim (or incremented if preexisting) 
## but can control the output directory  
##
## export G4DAWNFILE_DEST_DIR="/tmp/"   ## dawn needs trailing slash
## do not sully the normal environment with this ...
## set this as its needed

EOU
}



dawn-get(){

  n=$DAWN_NAME
  cd $LOCAL_BASE
  test -d dawn || ( sudo mkdir dawn && sudo chown $USER dawn )
  cd dawn 

  mkdir -p build bin
  cd build
  
  taz=$n.taz 
  url=$DAWN_URLBASE/$taz
 
  test -f $taz || curl -o $taz $url 
  test -d $DAWN_NAME || tar zxvf $taz

}


dawn-make(){

   cd $DAWN_BUILD
   make clean
   make guiclean

   ./configure    ## answer the questions... see ../../configure-dawn.txt

    make
	make install

}
