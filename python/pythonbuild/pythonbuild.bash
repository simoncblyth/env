
pythonbuild-usage(){

  cat << EOU
  
   separate off build related funcs for modularity ...
   
   
     PYTHON_NAME :  $PYTHON_NAME
   
     pythonbuild-get
         download and unpack 
         
     pythonbuild-configure
     pythonbuild-install
     pythonbuild-setuptools-get

     pythonbuild-dir     :  $(pythonbuild-dir)
     pythonbuild-prefix  :  $(pythonbuild-prefix)
                   installation prefix
     pythonbuild-cd      :
                    to \$(pythonbuild-dir)  
   
  
EOU

}



pythonbuild-env(){
   python-
}

pythonbuild-dir(){
  echo $SYSTEM_BASE/python/build/$PYTHON_NAME
}

pythonbuild-prefix(){
  echo $SYSTEM_BASE/python/$PYTHON_NAME 
}

pythonbuild-cd(){
   cd $(pythonbuild-dir)
}

pythonbuild-get(){

    local msg="=== $FUNCNAME :" 
	local nam=$PYTHON_NAME
	local tgz=$nam.tgz
    local ver=${nam/*-/}
    local url=http://www.python.org/ftp/python/$ver/$tgz

    local dir=$(dirname $(pythonbuild-dir))

    mkdir -p $dir
    cd $SYSTEM_BASE/python
    echo $msg nam $nam tgz $tgz ver $ver url $url dir $dir
 
    test -f $tgz || curl -L -O $url
    test -d build || mkdir build
    test -d build/$nam || tar -C build -zxvf $tgz 
}

pythonbuild-configure(){

	cd $(pythonbuild-dir)
	./configure --prefix=$(pythonbuild-prefix)
}

pythonbuild-install(){

    cd $(pythonbuild-dir)
	make
	make install
}

pythonbuild-setuptools-get(){

  [ "$PYTHON_HOME/bin" == $(dirname $(which python)) ] || ( echo your path to python is incorrect aborting && return  )

  cd $SYSTEM_BASE/python 
  test -f ez_setup.py || curl -O  http://peak.telecommunity.com/dist/ez_setup.py
  python  ez_setup.py

#
#Downloading http://cheeseshop.python.org/packages/2.5/s/setuptools/setuptools-0.6c5-py2.5.egg
#Processing setuptools-0.6c5-py2.5.egg
#Copying setuptools-0.6c5-py2.5.egg to /disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages
#Adding setuptools 0.6c5 to easy-install.pth file
#Installing easy_install script to /disk/d4/dayabay/local/python/Python-2.5.1/bin
#Installing easy_install-2.5 script to /disk/d4/dayabay/local/python/Python-2.5.1/bin
#
#Installed /disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/setuptools-0.6c5-py2.5.egg
#Processing dependencies for setuptools==0.6c5
#

   ## this puts easy_install in the PYTHON_HOME/bin
   which easy_install
 
}


