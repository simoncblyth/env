[ "$DYW_DBG" == "1" ] && echo $DYW_BASE/cernlib.bash


##
##
##  usage:
##
##     determine the appropriate settings from :
##         gcc -V 
##         uname -a   
##         open http://cernlib.web.cern.ch/cernlib/download/
##
##    G> p-cernlib   (transfer this file to target node)
##
##    P> ini
##    P> cernlib-get   (grab the binaries)
##
##

if [ "$NODE_TAG" == "L" ]; then

  CERNLIB_YEAR=2005
  CERNLIB_ARCH=slc4_amd64_gcc34

elif [ "$NODE_TAG" == "G" ]; then

  CERNLIB_YEAR=no-distro-for-mac-ppc
  CERNLIB_ARCH=no-distro-for-mac-ppc
 
elif ( [ "$NODE_TAG" == "G1" ] || [ "$NODE_TAG" == "P" ] || [ "$CLUSTER_TAG" == "$NODE_TAG" ] ) ; then

  CERNLIB_YEAR=2004
  CERNLIB_ARCH=slc3_ia32_gcc323

elif [ "$NODE_TAG" == "T" ]  ; then

  CERNLIB_YEAR=2004
  CERNLIB_ARCH=slc3_ia32_gcc323

elif [ "$NODE_TAG" == "N" ]; then

  echo cernlib setup externally 

else

  echo .bash_cernlib not setup for NODE_TAG $NODE_TAG CLUSTER_TAG $CLUSTER_TAG

fi  

CERNLIB_LEVEL=pro
CERNLIB_CMT="CERNLIB_prefix:$LOCAL_BASE/cernlib/$CERNLIB_YEAR CERNLIB_level:$CERNLIB_LEVEL"

export CERNLIB_CMT


cernlib-get(){

   y=$CERNLIB_YEAR
   a=$CERNLIB_ARCH
   l=$CERNLIB_LEVEL

   n=${y}_${a}

   cd $LOCAL_BASE
   test -d cernlib || ($SUDO mkdir cernlib && $SUDO chown $USER cernlib)
   cd cernlib && ( test -d $n || mkdir $n ) && cd $n 
   
   for t in cernbin cernlib 
   do
      tgz=$t.tar.gz
      test -f $tgz || curl -o $tgz http://cernlib.web.cern.ch/cernlib/download/$n/tar/$tgz 
	  tar -C ../ -zxvf $tgz
   done
	  
   ## the arch info is in the paths within the tarball, hence the -C ../	 

   ## huhh what needs /cern/pro ???
   [ "$LOCAL_NODE" == "pal" ] && ( test -d /cern || $SUDO mkdir /cern ) && ( cd /cern && $SUDO ln -s $LOCAL_BASE/cernlib/$y/$a $l )
   
}


