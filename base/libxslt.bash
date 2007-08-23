libxslt-env(){

  libxml2
  libxml2-env
  
  local vers
  if [ "$LIBXML2_VERSION" == "2.6.29" ]; then
     vers=1.1.21
  else
     echo no matched version 
     return 1 
  fi     
     
  export LIBXSLT_VERSION=$vers
  export LIBXSLT_NAME=libxslt-$vers
  export LIBXSLT_FOLD=$LOCAL_BASE/libxslt

}

libxslt-dir(){

    libxslt-env
    local dir=$LOCAL_BASE/libxslt/$LIBXSLT_NAME
    test -d $dir || ( echo error no folder $dir && return 1 ) 
    cd $dir
}



libxslt-get(){

    libxslt-env

    local dir=$LOCAL_BASE/libxslt
    [ -d $dir ] || ( sudo mkdir -p $dir && sudo chown $USER $dir ) 
   
    local name=$LIBXSLT_NAME
    local tgz=$name.tar.gz
    local url=ftp://xmlsoft.org/libxml2/$tgz
    
    cd $dir
    test -f $tgz || curl -o $tgz $url
    test -d $name || tar zxvf $tgz 
}



libxslt-configure(){

# http://jamesclarke.info/notes/libxml2

    libxslt-dir
   ./configure \
     --with-python=$PYTHON_HOME \
     --prefix=$LIBXSLT_FOLD \
     --with-libxml-prefix=$LIBXML2_FOLD \
     --with-libxml-include-prefix=$LIBXML2_FOLD/include \
     --with-libxml-libs-prefix=$LIBXML2_FOLD/lib \

}