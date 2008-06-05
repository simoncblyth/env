

#
#   http://docutils.sourceforge.net/
#  
#

docutils-get(){

   local nik=docutils
   local nam=$nik-snapshot
   local tgz=$nam.tgz
   local url=ftp://ftp.berlios.de/pub/docutils/$tgz
   local dir=$LOCAL_BASE/python/$nik
   local uir=$dir/$nik


   local iwd=$(pwd)
   
   mkdir -p $dir 
   cd $dir 
   test -f $tgz || curl -o $tgz $url 
   test -d $nik || tar zxvf $tgz

   cd $uir
   #cd $iwd


}

docutils-install(){

   python setup.py install

}

