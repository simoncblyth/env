
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
                 SVN_HOME    : $SVN_HOME
                 APACHE_HOME : $APACHE_HOME
                 SWIG_HOME   : $SWIG_HOME
                 PYTHON_HOME : $PYTHON_HOME

            Following recommendations in 
                $(svnbuild-dir)/subversion/bindings/swig/INSTALL


       svnbuild-krb-gssapi-kludge
             see svn/svn-build.bash for the details... does an inplace edit 
             of Makefile to avoid issue...
              
              ImportError: /disk/d4/dayabay/local/svn/subversion-1.4.0/lib/libsvn_ra_dav-1.so.0: undefined symbol: gss_delete_sec_context
              
 
       svnbuild-make
       svnbuild-install
              note this writes into APACHE as well
    
              suspect the below addition was made to httpd.conf
             
              # LoadModule foo_module modules/mod_foo.so
              LoadModule dav_svn_module     modules/mod_dav_svn.so
              LoadModule authz_svn_module   modules/mod_authz_svn.so
       
       
       svnbuild-swigpy
              NB must do after svnbuild-install and the LD_LIBRARY_PATH must be setup 
       
       svnbuild-pth
               writes $PYTHON_SITE/subversion.pth  

       svnbuild-swigpy-test
             

       svnbuild-again
              wipes and rebuilds the lot 



   issues ...
       
       python2.5
       apache2 version match  
       svn-bindings for python 2.5  
       
       subversion 1.4.2 is recommended  for Trac

EOU



}


svnbuild-env(){
   svn-
   apache-
   swig-
   python-
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
   
  cd  $(svnbuild-dir)
  ./configure  --prefix=$SVN_HOME --with-apxs=$APACHE_HOME/bin/apxs --with-swig=$SWIG_HOME/bin/swig PYTHON=$PYTHON_HOME/bin/python

  if [ "$NODE_TAG" == "P" ]; then
     svnbuild-kludge-py-bindings
  fi 


}


svnbuild-krb-gssapi-kludge(){

  ## needed on hfag+grid1 ? seems not on OSX
  
  cd $(svnbuild-dir)
  perl -pi.orig -e 's|^(SVN_APR_LIBS.*)$|$1 -L/usr/kerberos/lib -lgssapi_krb5|' Makefile
  diff Makefile{.orig,}
}




svnbuild-make(){
  cd  $(svnbuild-dir)
  $SUDO make
}

svnbuild-install(){
  cd  $(svnbuild-dir)
  $SUDO make install
}

svnbuild-swigpy(){
  cd $(svnbuild-dir)
  make swig-py
  make install-swig-py
}

svnbuild-pth(){	
  echo $SVN_HOME/lib/svn-python > $PYTHON_SITE/subversion.pth
}

svnbuild-swigpy-test(){

  python -c "from svn import client"
  
  python << EOT
from svn import core  
print (core.SVN_VER_MAJOR, core.SVN_VER_MINOR, core.SVN_VER_MICRO, core.SVN_VER_PATCH )
EOT

}


svnbuild-wipe(){
  cd $SYSTEM_BASE/svn
  [ -d build ] && rm -rf build
}

svnbuild-wipe-install(){
   
  cd $SYSTEM_BASE/svn
  [ "${SVN_NAME:0:3}" != "subversion" ] && echo bad name $SVN_NAME && return 1
  [ -d $SVN_NAME ] && rm -rf "$SVN_NAME"
}



svnbuild-again(){

   svnbuild-wipe
   svnbuild-wipe-install
   
   svnbuild-get
   svnbuild-configure
   svnbuild-make
   svnbuild-install
   svn-path      ## sets PATH and LD_LIBRARY_PATH needed by client builds
   
   svnbuild-swigpy
   svnbuild-pth
   
   svnbuild-swigpy-test

   

}


