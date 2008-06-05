#
#   issues:
#      1) logout from trac only works on firefox, where can drop the cookies?
#
#
#   modpython-x
#   modpython-i
#
#   modpython-apache2-configure
#   modpython-tracs-conf              spit config to stdout
#
#   modpython-get
#   modpython-configure
#   modpython-install
#
#  
#   http://www.modpython.org/live/mod_python-3.3.1/doc-html/inst-prerequisites.html
#   mod_python-3.3.1
#      Python 2.3.4 or later. Python versions less than 2.3 will not work.
#      Apache 2.0.54 or later. Apache versions 2.0.47 to 2.0.53 may work but have not been tested with this release. (For Apache 1.3.x, use mod_python version 2.7.x)
#
#
#
# APACHE2_NAME=httpd-2.0.59
# PYTHON_NAME=Python-2.5.1
#


modpython-env(){

   local-
   apache2-
   python-
   
   MODPYTHON_NAME=mod_python-3.3.1
   MODPYTHON_NIK=mod_python
   HOSTPORT=grid1.phys.ntu.edu.tw:6060

}



modpython-apache2-configure(){
  
   echo adding :  LoadModule python_module libexec/mod_python.so  to $APACHE2_CONF
   apache2-add-module python
   
}
   
#PythonPath 'sys.path'
#PythonPath "sys.path + ['/path/to/trac']"




modpython-get(){
  
  nam=$MODPYTHON_NAME
  nik=$MODPYTHON_NIK
  tgz=$nam.tgz

  url=http://apache.cdpa.nsysu.edu.tw/httpd/modpython/$tgz

  cd $LOCAL_BASE
  test -d $nik || ( $SUDO mkdir $nik && $SUDO chown $USER $nik )
  cd $nik

  test -f $tgz || curl -o $tgz $url
  test -d build || mkdir build
  test -d build/$nam || tar -C build -zxvf $tgz 

}

modpython-configure(){
    nam=$MODPYTHON_NAME
    nik=$MODPYTHON_NIK
	cd $LOCAL_BASE/$nik/build/$nam
	./configure -h
	
	if [ "$NODE_TAG" == "G" ]; then
	  ./configure --prefix=$LOCAL_BASE/$nik/$nam --with-apxs=/usr/sbin/apxs  --with-python=/usr/bin/python 
	  
	  ## --with-python-src=DIR	Path to python sources - required if you want to generate the documenation
	  ## so skip it   
 
	else
	  ./configure --prefix=$LOCAL_BASE/$nik/$nam --with-apxs=$APACHE2_HOME/sbin/apxs  --with-python=$PYTHON_HOME/bin/python --with-python-src=$PYTHON_HOME
    fi 
}

modpython-install(){
    nam=$MODPYTHON_NAME
    nik=$MODPYTHON_NIK
	cd $LOCAL_BASE/$nik/build/$nam
    make
	$SUDO make install

	ls -alst $APACHE2_HOME/libexec
}


# Leopard install against stock apache2 + python :
#
#	 installs mod_python.so into  /usr/libexec/apache2
#    creating /Library/Python/2.5/site-packages/mod_python
#  	  Writing /Library/Python/2.5/site-packages/mod_python-3.3.1-py2.5.egg-info
#










