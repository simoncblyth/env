
svnbuild-usage(){
 
  cat << EOU

  Start fresh ... as old svn-build is a bit of a morass of dependencies and env pollution...

    http://trac.edgewall.org/wiki/TracSubversion


       svnbuild-get  

   issues ...
       
       python2.5
       apache2 version match  
       svn-bindings for python 2.5  
       
       subversion 1.4.2 is recommended  for Trac



EOU



}


svnbuild-env(){

   elocal-
  
   export SVN_NAME=subversion-1.4.2
  

}


svnbuild-get(){

   local nam=$SVN_NAME
   local tgz=$nam.tar.gz
   local url=http://subversion.tigris.org/downloads/$tgz

   local dir=$SYSTEM_BASE/svn && mkdir -p $dir
   cd $dir
   
   [ ! -f $tgz ] && curl -O $url
   [ ! -d $nam ] && tar zxvf $tgz 


}



