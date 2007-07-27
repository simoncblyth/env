alias p-soxt="scp $HOME/.bash_soxt P:"
[ "$DYW_DBG" == "1" ] && echo $DYW_BASE/soxt.bash
##
##    This provides the glue between coin3d , openmotif and X11
##    allowing OpenInventor style 3D GUI to be used from X11 apps
##
##    Usage:
##            G> p-soxt
## 
##            P> ini
##            P> soxt-get
##            P> soxt-configure
##            P> soxt-edit-libtool
##            P> soxt-install
##
##


SOXT_NAME=SoXt-1.2.2

if ([ "$NODE_TAG" == "G1" ] || [ "$NODE_TAG" == "P" ] || [ "$NODE_TAG" == "$CLUSTER_TAG" ]); then
	SOXT_HOME=$LOCAL_BASE/soxt/$SOXT_NAME
else
	SOXT_HOME=$LOCAL_BASE/soxt
fi

SOXT_BUILD=$LOCAL_BASE/graphics/$SOXT_NAME


export SOXT_HOME 
export SOXT_BUILD


if [ "$CMTCONFIG" == "Darwin" ]; then
  export DYLD_LIBRARY_PATH=$SOXT_HOME/lib:${DYLD_LIBRARY_PATH}
else
  export   LD_LIBRARY_PATH=$SOXT_HOME/lib:${LD_LIBRARY_PATH}
fi

export OIVFLAGS="-I$COIN_HOME/include -I$SOXT_HOME/include"
export OIVLIBS="-L$SOXT_HOME/lib -lSoXt -L$COIN_HOME/lib -lCoin"


if [ "$CMTCONFIG" == "amd64_linux26" ]; then
   export OGLLIBS="-L/usr/X11R6/lib64 -lGLU -lGL"
else
   export OGLLIBS="-L/usr/X11R6/lib -lGLU -lGL"
fi

export ENV2GUI_VARLIST="OGLLIBS:SOXT_HOME:$ENV2GUI_VARLIST"




soxt-get(){

  n=$SOXT_NAME

  cd $LOCAL_BASE
  test -d graphics || ( $SUDO mkdir graphics && $SUDO chown $USER graphics )
  cd graphics
  
  tgz=$n.tar.gz
  url=http://ftp.coin3d.org/coin/src/all/$tgz

  test -d $n || ( curl -o $tgz $url && tar zxvf $tgz )

}


soxt-help(){

  cd $SOXT_BUILD || ( echo .bash_soxt ERROR must soxt-get first && exit ) 
  ./configure --help

}

soxt-configure(){


  test -d $COIN_HOME      || ( echo .bash_soxt must setup coin3d first && exit  )
  test -d $OPENMOTIF_HOME || ( echo .bash_soxt must setup openmotif first && exit  )
  test -d $SOXT_HOME      || ( mkdir -p $SOXT_HOME )
  
  cd $LOCAL_BADR
  test -d soxt || ( $SUDO mkdir soxt && $SUDO chown $USER soxt )

  cd $SOXT_BUILD || ( echo .bash_soxt ERROR must soxt-get first && exit ) 

  if [ "$LOCAL_NODE" == "g4pb" ]; then
     ./configure --with-motif=$OPENMOTIF_HOME --with-coin=$COIN_HOME --prefix=$SOXT_HOME --enable-darwin-x11
  else	  
     ./configure --with-motif=$OPENMOTIF_HOME --with-coin=$COIN_HOME --prefix=$SOXT_HOME
  fi


  #1st try ... suffer with multiple symbols in the link, so try without openmotif .. but no difference , so replace openmotif
  #./configure --with-motif=/usr/local/openmotif --with-coin=/usr/local/coin3d --prefix=/usr/local/soxt
  #./configure  --with-coin=/usr/local/coin3d --prefix=/usr/local/soxt

}


soxt-install(){
  cd $SOXT_BUILD || ( echo .bash_soxt ERROR must soxt-get and soxt-configure first && exit ) 
  make install
}


soxt-edit-libtool(){

  cd $SOXT_BUILD || ( echo .bash_soxt ERROR must soxt-get and soxt-configure first && exit ) 

  if [ "$CMTCONFIG" == "amd64_linux26" ]; then
     cp libtool libtool.original
     perl -pi -e 's|(predep_objects="/usr/lib/gcc/x86_64-redhat-linux/3.4.6/../../../../lib64/crti.o /usr/lib/gcc/x86_64-redhat-linux/3.4.6/crtbeginS.o")$|#SCB $1| ' libtool
     perl -pi -e 's|(postdep_objects="/usr/lib/gcc/x86_64-redhat-linux/3.4.6/crtendS.o /usr/lib/gcc/x86_64-redhat-linux/3.4.6/../../../../lib64/crtn.o")$|#SCB $1| ' libtool
  else
	 echo kludge not needed on $CMTCONFIG ???
  fi	 



#  GOT ERROR IN LINKING ...
#  /usr/lib/gcc/x86_64-redhat-linux/3.4.6/../../../../lib64/crti.o(.init+0x0):
#  In function `_init':: multiple definition of `_init'
#
#  TRIED, CONFIG WITHOUT openmotif... BUT GOT THE SAME
#
#  FOLLOWING TIP FROM THE coin3d FORUM http://www.coin3d.org/pipermail/coin-discuss/2005-April/005681.html
#
#  SOLVED BY COMMENTING OUT TWO LINES IN "libtool" FILE
#SCBpredep_objects="/usr/lib/gcc/x86_64-redhat-linux/3.4.6/../../../../lib64/crti.o /usr/lib/gcc/x86_64-redhat-linux/3.4.6/crtbeginS.o"
#SCB postdep_objects="/usr/lib/gcc/x86_64-redhat-linux/3.4.6/crtendS.o /usr/lib/gcc/x86_64-redhat-linux/3.4.6/../../../../lib64/crtn.o"
#
#

}



