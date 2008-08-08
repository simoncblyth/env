



apachebuild-env(){
   elocal-
   apache-
}

apachebuild-usage(){

  cat << EOU

   split off build related funcs for modularity 

      APACHE_NAME  :  $APACHE_NAME

      apachebuild-get
      apachebuild-configure
      apachebuild-install
               make and install
               
      apachebuild-cd
      apachebuild-dir  : $(apachebuild-dir)
      apachebuild-home : $(apachebuild-home)
     

 
      apachebuild-wipe  :
             deletes the build folder ... for a fresh start
      apachebuild-wipe-install
             wipes the $APACHE_NAME installation
             
      apachebuild-get
             unpacks again 
      apachebuild-buildconf 
             issue with modpython forcing to buildconf.. creating a new "configure"
             in order to use a newer libtool
  
      apachebuild-configure
      apachebuild-install

      $(type apachebuild-again)



EOU


}

apachebuild-get(){

  local nam=$APACHE_NAME
  local tgz=$nam.tar.gz
  local url=http://ftp.mirror.tw/pub/apache/httpd/$tgz

  cd $SYSTEM_BASE
  mkdir -p apache
  cd apache 
  
  [ ! -f $tgz ]   && curl -O $url
  mkdir -p build
  [ ! -d build/$nam ] && tar -C build -zxvf $tgz 
}


apachebuild-home(){
  #echo $SYSTEM_BASE/apache/$APACHE_NAME
  echo $APACHE_HOME
}

apachebuild-dir(){
   echo $SYSTEM_BASE/apache/build/$APACHE_NAME
}

apachebuild-cd(){
   cd $(apachebuild-dir)
}



## invoked with sudo bash -lc apache2-configure

apachebuild-configure(){
   cd $(apachebuild-dir)

   ##
   ## http://www.devshed.com/c/a/Apache/Building-Apache-the-Way-You-Want-It/3/
   ##
   ## ooops this does not include mod_proxy
   ## opts="--enable-mods-shared=most "
   ## local opts="--enable-mods-shared=all --enable-proxy=shared "

   # $ASUDO ./configure --prefix=$(apachebuild-home) --enable-modules=most --enable-shared --enable-so
   
   $SUDO ./configure --prefix=$(apachebuild-home) --enable-dav --enable-ssl
   
}

apachebuild-install(){
   cd $(apachebuild-dir)
   
   $SUDO make
   $SUDO make install

 

}

apachebuild-wipe(){
   local dir=$SYSTEM_BASE/apache
   [ ! -d $dir ] && return 0
   cd $dir
   [ -d build ] && rm -rf build
}

apachebuild-wipe-install(){
   local dir=$SYSTEM_BASE/apache
   [ ! -d $dir ] && return 0
   cd $dir
   [ "${APACHE_NAME:0:5}" != "httpd" ] && echo cannot proceed bad name $APACHE_NAME && return 1
   [ -d $APACHE_NAME ] && rm -rf $APACHE_NAME
}


apachebuild-buildconf(){

   cd $(apachebuild-dir)
   ./buildconf
   
}

apachebuild-again(){

   apachebuild-wipe
   apachebuild-wipe-install
   
   apachebuild-get
   apachebuild-configure
   apachebuild-install

   apacheconf-
   apacheconf-original---

}


