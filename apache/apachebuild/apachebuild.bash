
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

      apachebuild-dir  : $(apachebuild-dir)
      apachebuild-home : $(apachebuild-home)


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

   $ASUDO ./configure --prefix=$(apachebuild-home) --enable-modules=most --enable-shared=max 
}

apachebuild-install(){
   cd $(apachebuild-dir)
   
   $ASUDO make
   $ASUDO make install
}



