
pylibxml2-get(){


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
    local url= ftp://xmlsoft.org/libxml2/python/$tgz
   
    test -f $tgz || curl -o $tgz $url
    test -d $name || tar zxvf $tgz 


}