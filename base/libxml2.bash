
libxml2-env(){

  local vers=2.6.29
  export LIBXML2_VERSION=$vers
  export LIBXML2_NAME=libxml2-$vers

}


libxml2-get(){

    libxml2-env

    local dir=$LOCAL_BASE/libxml2
    [ -d $dir ] || ( sudo mkdir -p $dir && sudo chown $USER $dir ) 
   
    local name=$LIBXML2_NAME
    local tgz=$name.tar.gz
    local url=ftp://xmlsoft.org/libxml2/$tgz
    
    cd $dir
    test -f $tgz || curl -o $tgz $url
    test -d $name || tar zxvf $tgz 
}


libxml2-dir(){

    libxml2-env
    local dir=$LOCAL_BASE/libxml2/$LIBXML2_NAME
    test -d $dir || ( echo error no folder $dir && return 1 ) 
    cd $dir
}

libxml2-configure(){

    libxml2-dir
    ./configure --prefix=$LOCAL_BASE/libxml2 --with-python=$PYTHON_HOME

#
# checking for python... /data/usr/local/python/Python-2.5.1/bin/python
# Found Python version 2.5
# could not find python2.5/Python.h
#
#
# need to have the  --with-python=$PYTHON_HOME to find the python headers...
#
# Found python in /data/usr/local/python/Python-2.5.1/bin/python
# Found Python version 2.5
#

}




libxml2-make(){
  
    libxml2-dir 
    make
}

libxml2-install(){
  
    libxml2-dir 
    make install
}







libxml2-get-rpms(){

  ## not pursued ... rpms are linux specific so prefer not to follow this route 

    libxml2-env
     
    local vers=$LIBXML2_VERSION    
    local name=$LIBXML2_NAME
    local dame=libxml2-devel-$vers
    
    local srpm=$name-1.src.rpm
    local drpm=$dame-1.i386.rpm
    
    # there is no devel rpm for that version ... so try the tgz route 
    
    cd $dir

    local rpms="$srpm $drpm"
    for rpm in $rpms
    do
        
       local url=ftp://xmlsoft.org/libxml2/$rpm
       echo === libxml2-get $url to $rpm 
       test -f $rpm || curl -o $rpm $url
    done
    
  




}

