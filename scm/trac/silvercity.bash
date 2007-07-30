
#
#  another provider of syntax highlighting to trac...
#  installed principally to avoid the error messages in the trac.log and
#  associated performance hit
#
#      http://trac.edgewall.org/wiki/TracSyntaxColoring
#      http://trac.edgewall.org/wiki/SilverCity 
#       
#


silvercity-get(){

   local nik=silvercity
   local nam=SilverCity-0.9.7
   local tgz=$nam.tar.gz
   local url=http://nchc.dl.sourceforge.net/sourceforge/$nik/$tgz
   local dir=$LOCAL_BASE/python/$nik

   local iwd=$(pwd)
   
   mkdir -p $dir 
   cd $dir 
   test -f $tgz || curl -o $tgz $url 
   test -d $nam || tar zxvf $tgz


   cd $iwd
}

silvercity-install(){

   python setup.py install 
   # Writing /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/SilverCity-0.9.7-py2.5.egg-info

}

silvercity-test(){
   python -c "import SilverCity"
}


