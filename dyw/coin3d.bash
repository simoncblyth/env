[ "$DYW_DBG" == "1" ] && echo $DYW_BASE/coin3d.bash


##
##   
##   coin3d is an open source OpenInventor like implementation from SIM (Systems in Motion )
##   it together with the soxt glue allows X11 based programs to have the nice
##   OpenInventor 3D GUI
##
##  src and build positions between Darwin and AMD64 have diverged
##  but the install location is the same 
##
##    usage :
##          G> p-coin       ( transfer this script )
##          P> ini          ( pickup modified functions/env )
##
##          P> coin-get
##          P> coin-configure
##          P> coin-build
##
##

COIN_NAME=Coin-2.4.5

if ( [ "$NODE_TAG" == "G1" ] || [ "$NODE_TAG" == "P" ] || [ "$NODE_TAG" == "$CLUSTER_TAG" ]) ; then       
   COIN_HOME=$LOCAL_BASE/coin3d/$COIN_NAME
else 
   COIN_HOME=$LOCAL_BASE/coin3d
fi

ENV2GUI_VARLIST="COIN_HOME:$ENV2GUI_VARLIST"

export COIN_HOME
export ENV2GUI_VARLIST


if [ "$CMTCONFIG" == "Darwin" ]; then
   export DYLD_LIBRARY_PATH=$COIN_HOME/lib:${DYLD_LIBRARY_PATH}
else
   export   LD_LIBRARY_PATH=$COIN_HOME/lib:${LD_LIBRARY_PATH}
fi



coin-get(){

  n=$COIN_NAME

  cd $LOCAL_BASE
  test -d graphics || ( $SUDO mkdir graphics && $SUDO chown $USER graphics )
  cd graphics

  tgz=$n.tar.gz
  url=http://ftp.coin3d.org/coin/src/all/$tgz

  test -f $tgz || ( curl -o $tgz $url && tar zxvf $tgz )

}

coin-configure(){
	
  n=$COIN_NAME
  cd $LOCAL_BASE
  test -d graphics    || ( echo .bash_coin3d ... ERROR must coin-get 1st  && exit )
  test -d $COIN_HOME  ||  mkdir -p $COIN_HOME

  cd graphics
  mkdir -p ${n}-build
  cd ${n}-build

  if [ "$CMTCONFIG" == "Darwin" ]; then
     ../$n/configure --prefix=$COIN_HOME --without-framework --enable-darwin-x11 
  else	  
     ../$n/configure --prefix=$COIN_HOME
  fi

#************************* WARNING ****************************
#*
#* We have not tested Coin on the linux-gnu x86_64
#* platform with the g++ C++ compiler. Please report
#* back to us at <coin-support@coin3d.org> how it works out.
#*
#**************************************************************

}

coin-build(){

  n=$COIN_NAME
  cd $LOCAL_BASE/graphics/${n}-build || ( echo .bash_coin3d ... ERROR must coin-get and coin-configure first && exit ) 

  make 
  make install

}


