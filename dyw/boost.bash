

alias p-boost="scp $HOME/.bash_boost P:"

##
##  Usage:
##        boostjam-get
##        boostjam-check
##                          boostjam is a build system (like make) used to build the boost libraries
##        boost-get
##        boost-build
##                           complains re python configuration, but never mind...
##
##   Note:
##      the boost build instructions in 
##           open $BOOST_FOLDER/$BOOST_NAME/more/getting_started.html
##       recommend downloading a prebuilt boost.jam executable "bjam"     
##       from 
##            open http://sourceforge.net/project/showfiles.php?group_id=7586&package_id=72941
##
##

export BOOST_NAME="boost_1_33_1"
export BOOST_INAME="boost-1_33_1"
export BOOST_FOLDER="$LOCAL_BASE/boost"

if [ "$CMTCONFIG" == "Darwin" ]; then
   export BOOST_LIBTYPE="-d"
else
   export BOOST_LIBTYPE="-gcc"
fi

export BOOST_CMT="BOOST_lib_type:$BOOST_LIBTYPE BOOST_include_dir:$BOOST_FOLDER/include/$BOOST_INAME BOOST_library_dir:$BOOST_FOLDER/lib"


##
if [ "$LOCAL_NODE" == "g4pb" ]; then
  export BOOSTJAM_NAME="boost-jam-3.1.13-1-macosxppc"
elif [ "$LOCAL_NODE" == "pal" ]; then
  export BOOSTJAM_NAME="boost-jam-3.1.13-1-linuxx86"
else
  export BOOSTJAM_NAME="boost-jam-3.1.13-1-linuxx86"
fi


alias bjam=$LOCAL_BASE/boostjam/$BOOSTJAM_NAME/bjam



boostjam-get(){

   cd $LOCAL_BASE
   test -d boostjam || ( $SUDO mkdir boostjam && $SUDO chown $USER boostjam )
   cd boostjam 

   n=$BOOSTJAM_NAME
   tgz=$n.tgz
   url=http://jaist.dl.sourceforge.net/sourceforge/boost/$tgz

   test -f $tgz || curl -o $tgz $url
   test -d $n   || tar zxvf $tgz 
   
}

boostjam-check(){
   which bjam
   bjam -v
}




boost-get(){

  n=$BOOST_NAME
  cd $LOCAL_BASE
  test -d boost || ( $SUDO mkdir boost && $SUDO chown $USER boost )
  cd boost

  tgz=$n.tar.gz  
  url=http://jaist.dl.sourceforge.net/sourceforge/boost/$tgz
  
  test -f $tgz || curl -o $tgz $url
  test -d $n   || tar zxvf $tgz

}




boost-build(){

  n=$BOOST_NAME
  cd $LOCAL_BASE/boost/$n

  if [ "$LOCAL_NODE" == "g4pb" ]; then
	  
     bjam "-sTOOLS=darwin" --prefix=$LOCAL_BASE/boost  install
   
  elif [ "$LOCAL_NODE" == "pal" ]; then
  
     bjam "-sTOOLS=gcc" --prefix=$LOCAL_BASE/boost  install

  else
	  
     bjam "-sTOOLS=gcc" --prefix=$LOCAL_BASE/boost  install
	
  fi

}
