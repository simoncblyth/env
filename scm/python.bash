
#
#    python-x
#    python-i
#
#    python-get
#    python-configure 
#    python-install
#
#    python-setuptools-get
#    python-pygments-get
#    python-crack-egg
#


python-x(){  scp $SCM_HOME/python.bash ${1:-$TARGET_TAG}:$SCM_BASE ; }
python-i(){ . $SCM_HOME/python.bash ; }



python-mac-check(){

   find $PYTHON_SITE -name '*.so' -exec otool -L {} \; | grep ython

}

#
#   reveals that "libsvn" is hooked up to the wrong python ....
#
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/libsvn/_client.so: /System/Library/Frameworks/Python.framework/Versions/2.3/Python (compatibility version 2.3.0, current version 2.3.5)
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/libsvn/_core.so:   /System/Library/Frameworks/Python.framework/Versions/2.3/Python (compatibility version 2.3.0, current version 2.3.5)
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/libsvn/_delta.so:  /System/Library/Frameworks/Python.framework/Versions/2.3/Python (compatibility version 2.3.0, current version 2.3.5)
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/libsvn/_fs.so:     /System/Library/Frameworks/Python.framework/Versions/2.3/Python (compatibility version 2.3.0, current version 2.3.5)
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/libsvn/_ra.so:     /System/Library/Frameworks/Python.framework/Versions/2.3/Python (compatibility version 2.3.0, current version 2.3.5)
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/libsvn/_repos.so:  /System/Library/Frameworks/Python.framework/Versions/2.3/Python (compatibility version 2.3.0, current version 2.3.5)
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/libsvn/_wc.so:     /System/Library/Frameworks/Python.framework/Versions/2.3/Python (compatibility version 2.3.0, current version 2.3.5)
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/mod_python/_psp.so:
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/neo_cgi.so:
# /usr/local/python/Python-2.5.1/lib/python2.5/site-packages/pysqlite2/_sqlite.so:
#


python-get(){

	nam=$PYTHON_NAME
	tgz=$nam.tgz
    url=http://www.python.org/ftp/python/2.5.1/$tgz

    cd $LOCAL_BASE
	test -d python || ( $SUDO mkdir python && $SUDO chown $USER python )
	cd python

    test -f $tgz || curl -o $tgz $url
    test -d build || mkdir build
    test -d build/$nam || tar -C build -zxvf $tgz 
}

python-configure(){

	nam=$PYTHON_NAME
	cd $LOCAL_BASE/python/build/$nam
	./configure --prefix=$LOCAL_BASE/python/$nam
}

python-install(){

	nam=$PYTHON_NAME
	cd $LOCAL_BASE/python/build/$nam
	make
	make install
}

python-setuptools-get(){

  [ "$PYTHON_HOME/bin" == $(dirname $(which python)) ] || ( echo your path to python is incorrect aborting && return  )

  setupdir=$LOCAL_BASE/python/setuptools 
  ezsetup=$setupdir/ez_setup.py
  mkdir -p $setupdir 
  test -f $ezsetup || curl -o $ezsetup  http://peak.telecommunity.com/dist/ez_setup.py

  cd $setupdir
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


python-pygments-get(){
	
   pygments_dir=$LOCAL_BASE/python/pygments
   pygments_egg=Pygments-0.7.1-py2.5.egg
   mkdir -p $pygments_dir && cd $pygments_dir
   test -f  $pygments_egg  || curl -o $pygments_egg http://jaist.dl.sourceforge.net/sourceforge/pygments/$pygments_egg
   $SUDO easy_install $pygments_egg 

#Processing Pygments-0.7.1-py2.5.egg
#creating /disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/Pygments-0.7.1-py2.5.egg
#Extracting Pygments-0.7.1-py2.5.egg to /disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages
#Adding Pygments 0.7.1 to easy-install.pth file
#Installing pygmentize script to /disk/d4/dayabay/local/python/Python-2.5.1/bin
#
#Installed /disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/Pygments-0.7.1-py2.5.egg
#Processing dependencies for Pygments==0.7.1
#

}




python-crack-egg(){

  path=${1:-dummy.egg}

  [ -f "$path" ] || ( echo the path $path doesnt correspond to a file && return 1 )
  [ -d "$path" ] && ( echo the egg $path is cracked already           && return 1 )

  cd $(dirname $path)
  base=$(basename $path)
  
  sudo mv $base $base.zip
  sudo mkdir $base 
  cd $base
  sudo unzip ../$base.zip
  sudo rm ../$base.zip 
  

}

