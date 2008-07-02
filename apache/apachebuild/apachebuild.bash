
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


EOU


}

apachebuild-get(){

  local nam=$APACHE_NAME
  local tgz=$nam.tar.gz
  local url=http://ftp.mirror.tw/pub/apache/httpd/$tgz

  cd $SYSTEM_BASE
  test -d apache 
  cd apache 
  
  [ ! -f $tgz ]   && curl -O $url
  mkdir -p build
  [ ! -d build/$nam ] && tar -C build -zxvf $tgz 
}


## invoked with sudo bash -lc apache2-configure

apachebuild-configure(){
   cd $APACHE2_BUILD

   ##
   ## http://www.devshed.com/c/a/Apache/Building-Apache-the-Way-You-Want-It/3/
   ##

   ## ooops this does not include mod_proxy
   ##opts="--enable-mods-shared=most "
   opts="--enable-mods-shared=all --enable-proxy=shared "


   $ASUDO ./configure --help
   layout="--prefix=$APACHE2_HOME --enable-layout=GNU "

   $ASUDO ./configure ${layout} ${opts}
}

apachebuild-install(){
   cd $APACHE2_BUILD
   $ASUDO make
   $ASUDO make install
}



