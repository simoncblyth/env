
alias p-xercesc="scp $HOME/.bash_xercesc P:"
[ "$DYW_DBG" == "1" ] && echo $DYW_BASE/xercesc.bash
##
## Usage :
##        xercesc-get
##        xercesc-build
##
##


XERCESC_NAME="xerces-c-src_2_7_0"
export XERCESCROOT="$LOCAL_BASE/xercesc/$XERCESC_NAME"
export XERCESC_CMT="XERCES_prefix:$XERCESCROOT"
export ENV2GUI_VARLIST="XERCESCROOT:$ENV2GUI_VARLIST"

if [ "$CMTCONFIG" == "Darwin" ]; then
   export DYLD_LIBRARY_PATH=$XERCESCROOT/lib:$DYLD_LIBRARY_PATH
else
   export   LD_LIBRARY_PATH=$XERCESCROOT/lib:$LD_LIBRARY_PATH
fi

xercesc-usage(){ cat << EOU

EOU
}


xercesc-get(){

  n=$XERCESC_NAME
  cd $LOCAL_BASE
  test -d xercesc || ( $SUDO mkdir xercesc && $SUDO chown $USER xercesc )
  cd xercesc

  tgz=xerces-c-current.tar.gz
  test -f $tgz || curl -o $tgz http://www.apache.org/dist/xml/xerces-c/$tgz
  test -d $n  || tar zxvf $tgz 

}


xercesc-build(){


  ## instructions   file:///usr/local/xercesc/xerces-c-src_2_7_0/doc/html/install.html

  ## build the lib
  cd $XERCESCROOT/src/xercesc ; 

  if [ "$LOCAL_NODE" == "g4pb" ]; then
	  
	arch=macosx 
    ./runConfigure -p $arch  -n native -t native
    make

    ## build samples
    cd $XERCESCROOT/samples
    ./runConfigure -p $arch 
    make
  
    ## run a sample  (the library path must be set)
    cd $XERCESCROOT/bin
    ./CreateDOMDocument
	
  make
	
  elif [ "$LOCAL_NODE" == "pal" ]; then

    arch=linux
    ## get incompatible libs in link ... but still succeeds	, when dont specify -b 64
 
    ./runConfigure -p $arch -b 64
    make
  
    ## build samples
    cd $XERCESCROOT/samples
    ./runConfigure -p $arch -b 64
    make
  
    ## run a sample  (the library path must be set)
    cd $XERCESCROOT/bin
    ./CreateDOMDocument
	
  else

   arch=linux
 
    ./runConfigure -p $arch 
    make
  
    ## build samples
    cd $XERCESCROOT/samples
    ./runConfigure -p $arch 
    make
  
    ## run a sample  (the library path must be set)
    cd $XERCESCROOT/bin
    ./CreateDOMDocument
	
  fi
}



