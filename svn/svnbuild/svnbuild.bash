
svnbuild-usage(){
 
  cat << EOU

   Start fresh ... as old svn-build is a bit of a morass of dependencies and env pollution...
      http://trac.edgewall.org/wiki/TracSubversion


       svnbuild-get   :
           gets and unpacks both the subversion tgz and the -deps which
           explode ontop of the primary tgz adding folders:  neon,apr,apr-util and zlib
       
       svnbuild-dir   :  $(svnbuild-dir)  

       svnbuild-configure :
            configure depends on ...
                 APACHE_HOME : $APACHE_HOME
                 SWIG_HOME   : $SWIG_HOME
                 PYTHON_HOME : $PYTHON_HOME




   issues ...
       
       python2.5
       apache2 version match  
       svn-bindings for python 2.5  
       
       subversion 1.4.2 is recommended  for Trac

EOU



}


svnbuild-env(){

   elocal-
   svn-
  
}


svnbuild-get-(){

   local nam=$1
   local tgz=$nam.tar.gz
   local url=http://subversion.tigris.org/downloads/$tgz

   local dir=$SYSTEM_BASE/svn && mkdir -p $dir
   cd $dir
   
   [ ! -f $tgz ] && curl -O $url
   mkdir -p build
   [ ! -d build/$nam ] && tar -C build -zxvf $tgz 

}

svnbuild-get(){
   svnbuild-get- $SVN_NAME
   svnbuild-get- $SVN_NAME2
}





svnbuild-dir(){
   echo $SYSTEM_BASE/svn/build/$SVN_NAME
}

svnbuild-cd(){
   cd $(svnbuild-dir)
}


svnbuild-configure(){

   
  ./configure  --prefix=$SVN_HOME --with-apxs=$APACHE_HOME/sbin/apxs --with-swig=$SWIG_HOME/bin/swig PYTHON=$PYTHON_HOME/bin/python

  #  speifying PYTHON on the configure commandline is recommended in $SVN_BUILD/subversion/bindings/swig/INSTALL
#


}

