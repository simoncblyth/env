
pylibxml2-get(){

  #  did not pursue this method ... attempting to use the inplace libxml2 ..
  #  but rather the py bindings were build while building libxml2 itself
  #

    local dir=$LOCAL_BASE/python/pylibxml2

   # 
   # http://xmlsoft.org/XSLT/python.html 
   #
   # on hfag xsltproc --version indicates  2.6.16 is present
   #  ... but no matching distro
   #         nearest are  2.6.15 and 2.6.21  
   #
   
    local name=libxml2-python-2.6.21
    mkdir -p $dir
    cd $dir
    
    local tgz=$name.tar.gz
    local url=ftp://xmlsoft.org/libxml2/python/$tgz
   
    test -f $tgz || curl -o $tgz $url
    test -d $name || tar zxvf $tgz 

   #
   # python setup.py --help
   # would suggest that the version of libxml2 on hfag is too old to be usable with the python in use 2.5.1
   # 

}