

modpython-usage(){


    cat << EOU
    
     http://www.modpython.org/live/mod_python-3.3.1/doc-html/inst-prerequisites.html
     http://www.modpython.org/live/mod_python-3.3.1/doc-html/modpython.html

     mod_python-3.3.1
         Python 2.3.4 or later. Python versions less than 2.3 will not work.
         Apache 2.0.54 or later. Apache versions 2.0.47 to 2.0.53 may work but have not been 
         tested with this release. (For Apache 1.3.x, use mod_python version 2.7.x)    
    
    
             MODPYTHON_NAME : $MODPYTHON_NAME
             MODPYTHON_HOME : $MODPYTHON_HOME 
             APACHE_NAME    : $APACHE_NAME
             APACHE_HOME    : $APACHE_HOME
             PYTHON_NAME    : $PYTHON_NAME
             PYTHON_HOME    : $PYTHON_HOME
             SUDO           : $SUDO
    
        modpython-dir  : $(modpython-dir)
    
    
        modpython-get
        modpython-configure
        modpython-install

     missing mod_python.so ...
          http://www.modpython.org/FAQ/faqw.py?req=show&file=faq04.001.htp
 
        after rebuilding apache with the newer libtool...
 
        modpython-wipe
               delete the build 
        modpython-wipe-install
               delete the install dir name $MODPYTHON_NAME        
               
        modpython-get
               unpack again
               
        modpython-configure
        modpython-install
        
        
        
     do it all again from scratch    
        $(type modpython-again)

 

EOU

}


modpython-notes(){

cat << EON

Performing DSO installation.

/usr/bin/install -c -d /data/env/system/apache/httpd-2.0.63/modules
/usr/bin/install -c src/mod_python.so /data/env/system/apache/httpd-2.0.63/modules
/usr/bin/install: cannot stat `src/mod_python.so': No such file or directory
make[1]: *** [install_dso] Error 1
make[1]: Leaving directory `/data/env/system/mod_python/build/mod_python-3.3.1'
make[1]: Entering directory `/data/env/system/mod_python/build/mod_python-3.3.1'
cd dist && make install_py_lib
make[2]: Entering directory `/data/env/system/mod_python/build/mod_python-3.3.1/dist'
make[3]: Entering directory `/data/env/system/mod_python/build/mod_python-3.3.1/src'
make[3]: `psp_parser.c' is up to date.
make[3]: Leaving directory `/data/env/system/mod_python/build/mod_python-3.3.1/src'
if test -z "" ; then \
        /data/env/system/python/Python-2.5.1/bin/python setup.py install --optimize 2 --force ; \
else \
        /data/env/system/python/Python-2.5.1/bin/python setup.py install --optimize 2 --force --root  ; \
fi

EON

}




modpython-env(){

   elocal-   
   apache-
   python-
   
   export MODPYTHON_NAME=mod_python-3.3.1
   export MODPYTHON_HOME=$SYSTEM_BASE/mod_python/$MODPYTHON_NAME
}

modpython-get(){
  
  local nam=$MODPYTHON_NAME
  local tgz=$nam.tgz
  local url=http://apache.cdpa.nsysu.edu.tw/httpd/modpython/$tgz

  cd $SYSTEM_BASE
  mkdir -p mod_python
  
  cd mod_python

  [ ! -f $tgz ] && curl -O $url
  mkdir -p build 
  [ ! -d build/$nam ] && tar -C build -zxvf $tgz 

}

modpython-dir(){
  echo $SYSTEM_BASE/mod_python/build/$MODPYTHON_NAME
}


modpython-configure(){

    cd $(modpython-dir)

	./configure -h
	
	if [ "$NODE_TAG" == "G" ]; then
	  ./configure --prefix=$MODPYTHON_HOME --with-apxs=/usr/sbin/apxs  --with-python=/usr/bin/python 
	  ## --with-python-src=DIR	Path to python sources - required if you want to generate the documenation so skip it   
 
	else
	  ./configure --prefix=$MODPYTHON_HOME --with-apxs=$APACHE_HOME/bin/apxs  --with-python=$PYTHON_HOME/bin/python 
      ## --with-python-src=$PYTHON_HOME
    fi 
}

modpython-install(){
   
    cd $(modpython-dir)
    make
	$SUDO make install

}



modpython-wipe(){

  cd $SYSTEM_BASE/mod_python
  rm -rf build

}


modpython-wipe-install(){

  cd $SYSTEM_BASE/mod_python
  
  [ "${MODPYTHON_NAME:0:10}" != "mod_python" ] && echo bad name $MODPYTHON_NAME cannot proceed && return 1
  
  rm -rf $MODPYTHON_NAME

}


modpython-again(){

    modpython-wipe
    modpython-wipe-install
    
    modpython-get
    modpython-configure
    modpython-install


}







#
#
#modpython-apache2-configure(){
#  
#   echo adding :  LoadModule python_module libexec/mod_python.so  to $APACHE2_CONF
#   apache2-add-module python
#   
#}
#   
#PythonPath 'sys.path'
#PythonPath "sys.path + ['/path/to/trac']"
#
#
#
#
# Leopard install against stock apache2 + python :
#
#	 installs mod_python.so into  /usr/libexec/apache2
#    creating /Library/Python/2.5/site-packages/mod_python
#  	  Writing /Library/Python/2.5/site-packages/mod_python-3.3.1-py2.5.egg-info
#










