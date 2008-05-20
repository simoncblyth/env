#
#
#      svn-get
#      svn-deps-get
#      svn-wipe
#      svn-configure                   needs APACHE2_HOME SWIG_HOME PYTHON_HOME
#      svn-kludge-py-bindings          edits SVN_APR_LIBS in $SVN_BUILD/Makefile adding gssapi_krb5
#      svn-make                 
#      svn-install              
#      svn-check                      <== very slow on hfag
#
#
#  ... python bindings ....
#
#      svn-swig-readme
#      svn-install-py-bindings
#      svn-copy-py-bindings-to-site-packages      python then knows about the svn package , without PYTHONPATH 
#      svn-test-py-bindings      
#      svn-ldd                                     library debugging
#
#
#
#
#

svn-build-env(){

   elocal-
   svn-
   apache2-
   swig- 
   python-

   
   svn-build-base

}


svn-build-base(){

   local SVN_NAME2=subversion-deps-1.4.0
   local SVN_URLBASE=http://subversion.tigris.org/downloads 

}


svn-build-all(){

  svn-build-wipe
  svn-build-get
  svn-build-deps-get
  svn-build-configure
  

  svn-build-make
  svn-build-install
  svn-build-check

  ## svn-build-kludge-py-bindings         needed on hfag+grid1 
  svn-build-install-py-bindings

  ## svn-build-copy-py-bindings-to-site-packages      replaced by svn-pth-connect
  svn-build-pth-connect  
  
  svn-build-test-py-bindings

}

svn-build-wipe(){

  n=$SVN_NAME
  nik=$SVN_ABBREV
  cd $LOCAL_BASE/$nik

  rm -rf build/$n 
}


svn-build-get(){

  n=$SVN_NAME
  tgz=$n.tar.gz
  url=$SVN_URLBASE/$tgz
  nik=$SVN_ABBREV

  cd $LOCAL_BASE

  test -d $nik || ( $SUDO mkdir $nik && $SUDO chown $USER $nik )
  cd $nik
  
  test -f $tgz || curl -o $tgz $url
  test -d build || mkdir build
  test -d build/$n || tar -C build -zxvf $tgz 
}


svn-build-deps-get(){

  n=$SVN_NAME2
  tgz=$n.tar.gz
  url=$SVN_URLBASE/$tgz
  nik=$SVN_ABBREV

  cd $LOCAL_BASE
  
  test -d $nik || ( $SUDO mkdir $nik && $SUDO chown $USER $nik )
  cd $nik
  
  test -f $tgz || curl -o $tgz $url
  test -d build || mkdir build
  test -d build/$SVN_NAME && tar -C build -zxvf $tgz 
 
 ## this is layered on top of the svn distro so cannot to an analogous test 

}



svn-build-configure(){

  cd $SVN_BUILD

  layout="--prefix=$SVN_HOME "
  ./configure -h
  #./configure  $layout --with-apxs=$APACHE2_HOME/sbin/apxs --with-swig=$SWIG_HOME/bin/swig PYTHON=$PYTHON_HOME/bin/python
  ./configure  $layout --with-apxs=$APACHE2_HOME/sbin/apxs --with-swig=$SWIG_HOME/bin/swig PYTHON=$PYTHON_HOME/bin/python

#  speifying PYTHON on the configure commandline is recommended in $SVN_BUILD/subversion/bindings/swig/INSTALL
#
#
# ==============> check at the end of the configure about regards python extensions, ensure are linking again the desired python
# 
# checking for inflate in -lz... yes
# checking for /usr/local/python/Python-2.5.1/bin/python... /usr/local/python/Python-2.5.1/bin/python
# checking for JDK... yes
# checking for perl... /usr/bin/perl
# checking for ruby... /usr/bin/ruby
# can't find header files for ruby.
# configure: WARNING: The detected Ruby is too old for Subversion to use
# configure: WARNING: A Ruby which has rb_hash_foreach is required to use the
# configure: WARNING: Subversion Ruby bindings
# configure: WARNING: Upgrade to the official 1.8.2 release, or later
# checking swig version... 1.3.29
# configure: Configuring python swig binding
# checking for Python includes... -I/usr/local/python/Python-2.5.1/include/python2.5
# checking for compiling Python extensions... gcc -fno-strict-aliasing -Wno-long-double -no-cpp-precomp -mno-fused-madd -DNDEBUG -g -O3 -Wall -Wstrict-prototypes 
# checking for linking Python extensions... gcc -bundle -undefined dynamic_lookup -bundle_loader /usr/local/python/Python-2.5.1/bin/python
# checking for linking Python libraries... -bundle -undefined dynamic_lookup -bundle_loader /usr/local/python/Python-2.5.1/bin/python
# 
# 

}

svn-build-kludge-py-bindings(){

  ## needed on hfag+grid1 ? seems not on OSX

  cd $SVN_BUILD
  perl -pi.orig -e 's|^(SVN_APR_LIBS.*)$|$1 -L/usr/kerberos/lib -lgssapi_krb5|' Makefile
  diff Makefile{.orig,}
}

#
#  http://trac.edgewall.org/ticket/3706
#
#
# [dayabaysoft@grid1 subversion-1.4.0]$ which neon-config
#  /disk/d4/dayabay/local/svn/subversion-1.4.0/bin/neon-config
#
#
# [dayabaysoft@grid1 subversion-1.4.0]$ neon-config --libs
#   -L/disk/d4/dayabay/local/svn/subversion-1.4.0/lib -lneon -lz
#   -L/usr/kerberos/lib -lgssapi_krb5 -lkrb5 -lk5crypto -lcom_err -lexpat
#
#           maybe there will be kerberos issues down the road ??
#
#
#  http://www.liucougar.net/blog/archives/date/2006/10/
#
# When setuping trac on a server, I encountered a strange issue: the python
# binding of subversion can not be loaded by python, I got something like this:
#
# libsvn_ra_dav-1.so.0: undefined symbol: gss_delete_sec_context
# After googling around, I found out that, someone suggested to disable neon
# when compiling subversion 1.4. That did do the trick, and binding works.
# However, with neon disabled, svn can not work with http/https repositories,
# which is not acceptable. So I have to find another workaround.
#
# Google told me that gss_delete_sec_context is part of libgssapi.so, and "ldd
# libsvn_ra_dav-1.so.0" reveals that it does not link to any libgssapi.so at
# all. The obvious workaround is to explicitly specify that in the Makefile.
#
# Edit the top Makefile in subversion 1.4.0, append "-lgssapi" to this line:
#
# SVN_APR_LIBS = ...
# ( is the actual content you will see) after installing it, everything works
# fine now.
#
# It may be argued that this is not a subversion issue, rather than a neon one
# (it should link to gssapi). As subversion uses a bundled neon, so maybe it is
# more faire to call it a subversion bug.
#

svn-build-make(){
  cd $SVN_BUILD
  make
}


svn-build-install(){

  cd $SVN_BUILD
  $SUDO make install

# NB this adds a couple of mod_{authz,dav}_svn.so to the apache2 modules $APACHE2_HOME/libexec/
#
#   on 
#
# cp .libs/mod_dav_svn.so /data/usr/local/apache2/httpd-2.0.59/libexec/mod_dav_svn.so
# cp: cannot create regular file `/data/usr/local/apache2/httpd-2.0.59/libexec/mod_dav_svn.so': Permission denied
# apxs:Error: Command failed with rc=65536
#  
}





svn-build-check(){

  cd $SVN_BUILD
  make check

# huh.... seems very slow on hfag ???
#
#  on hfag...
#
# Running all tests in authz_tests.py...success
# At least one test was SKIPPED, checking /data/usr/local/svn/build/subversion-1.4.0/tests.log
# SKIP:  utf8_tests.py 1: conversion of paths and logs to/from utf8
# SKIP:  svnsync_tests.py 14: verify that unreadable content is not synced
# SKIP:  svnsync_tests.py 15: verify that copies from unreadable dirs work
# SKIP:  authz_tests.py 1: authz issue #2486 - open root
# SKIP:  authz_tests.py 2: authz issue #2486 - open directory
# SKIP:  authz_tests.py 3: broken authz files cause errors
# SKIP:  authz_tests.py 4: test authz for read operations
# SKIP:  authz_tests.py 5: test authz for write operations
# SKIP:  authz_tests.py 6: test authz for checkout
# SKIP:  authz_tests.py 7: test authz for log and tracing path changes
# SKIP:  authz_tests.py 8: test authz for checkout and update
# SKIP:  authz_tests.py 9: test authz for export with unreadable subfolder
#
#
#
#  on g4pb get many failures...  $SVN_BUILD/tests.log
#
#
#


}



#
# libtool: link: warning: `/usr/lib/gcc-lib/i386-redhat-linux/3.2.3/../../..//libexpat.la' seems to be moved
#
#
# many failures from the checks
# At least one test FAILED, checking /disk/d4/dayabay/local/svn/build/subversion-1.4.0/tests.log
#
#  presumably as the test repository the tests setup are on a network share disk that doesnt
#  meet the requirements for locking etc..
#

svn-build-swig-readme(){
  cd $SVN_BUILD/subversion/bindings/swig
  cat INSTALL 
}








svn-build-install-py-bindings(){
#
#    If Subversion was already installed without the SWIG bindings, on Unix you'll need to re-configure Subversion 
#    and make swig-py, make install-swig-py
#  see  $SVN_BUILD/subversion/bindings/swig/INSTALL
#
  cd $SVN_BUILD
  make swig-py
  make install-swig-py
}


svn-build-pth-connect(){
	
 echo $SVN_HOME/lib/svn-python > $PYTHON_SITE/subversion.pth

}


svn-build-copy-py-bindings-to-site-packages(){

  echo this is replaced by svn-pth-connect

  ##  this is done by the install-py-bindings 
  ##
  #cp -r $SVN_HOME/lib/svn-python/svn    $PYTHON_HOME/lib/python2.5/site-packages/
  #cp -r $SVN_HOME/lib/svn-python/libsvn $PYTHON_HOME/lib/python2.5/site-packages/
}



svn-build-test-py-bindings(){

  which python

  python << EOT
from svn import core  
print (core.SVN_VER_MAJOR, core.SVN_VER_MINOR, core.SVN_VER_MICRO, core.SVN_VER_PATCH )
EOT
#
#  aiming for :
# (1, 4, 0, 0)

  python -c "from svn import client"

#  no output is success
#
#
#
#  when have issues with the bindings :
#
#[dayabaysoft@grid1 subversion-1.4.0]$ python Python 2.5.1 (r251:54863, Apr 24 2007, 17:01:00) 
#[GCC 3.2.3 20030502 (Red Hat Linux 3.2.3-58)] on linux2
#Type "help", "copyright", "credits" or "license" for more information.
#>>> from svn import core Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
#    File
#	"/disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/svn/core.py",
#	line 19, in <module>
#	    from libsvn.core import *
#		  File
#		  "/disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/libsvn/core.py",
#		  line 5, in <module>
#		      import _core
#			  ImportError:
#			  /disk/d4/dayabay/local/svn/subversion-1.4.0/lib/libsvn_ra_dav-1.so.0:
#			  undefined symbol: gss_delete_sec_context
#			  >>> 
#
#

}

svn-build-ldd(){
   ldd $(which svn)
   ldd $(which python)
   ldd $APACHE2_HOME/sbin/httpd
}






