#
#
#  clearsilver-get
#  clearsilver-wipe 
#  clearsilver-configure
#  clearsilver-install
#
#



CLEARSILVER_NAME=clearsilver-0.10.4
CLEARSILVER_NIK=clearsilver
export CLEARSILVER_HOME=$LOCAL_BASE/$CLEARSILVER_NIK/$CLEARSILVER_NAME


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
   
   nam=$CLEARSILVER_NAME
   nik=$CLEARSILVER_NIK
   
   cd $LOCAL_BASE/$nik/build/$nam
   ./configure --prefix=$CLEARSILVER_HOME  --with-python=$PYTHON_HOME/bin/python --enable-python --disable-ruby --disable-perl --disable-apache --disable-csharp --disable-java
}

clearsilver-install(){

   nam=$CLEARSILVER_NAME
   nik=$CLEARSILVER_NIK
   
   cd $LOCAL_BASE/$nik/build/$nam

   make
   make install

   ## this places a library at :  $PYTHON_HOME/lib/python2.5/site-packages/neo_cgi.so

}




