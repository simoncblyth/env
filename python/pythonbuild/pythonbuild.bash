pythonbuild-src(){    echo python/pythonbuild/pythonbuild.bash ; }
pythonbuild-source(){ echo ${BASH_SOURCE:-$ENV_HOME/$(pythonbuild-src)} ; }
pythonbuild-vi(){     vi $(pythonbuild-source) ; }
pythonbuild-usage(){ cat << EOU
 
pythonbuild-
=============

Build related funcs for modularity ...
   
   
pythonbuild-dir    
     emit build dir 

pythonbuild-prefix  
     emit prefix 

pythonbuild-get
     download and unpack 

pythonbuild-configure
     note the the --enable-shared is required to create ./Python-2.5.1/lib/libpython2.5.so

pythonbuild-install
     build and install into the prefixed dir
     
pythonbuild-wipe  :
     delete the build dir

pythonbuild-wipe-install 
     delete the install dir called $PYTHON_NAME
    
pythonbuild-solibfix
     place a link in python config dir to the libpython2.5.so two levels up
     to keep modpython nice an slim.
     http://code.google.com/p/modwsgi/wiki/InstallationIssues
    
pythonbuild-again
     wipe and rebuild
                
pythonbuild-setuptools-get 
     some extras

  
sudo python chestnut via LD_RUN_PATH
--------------------------------------- 
    
To avoid a common problem : "sudo python" not finding libpython2.5.so 
when the python is in a different place than the system python...
due to the LD_LIBRARY_PATH not being set in root users environment , can use LD_RUN_PATH when 
building python according to http://www.modpython.org/pipermail/mod_python/2007-March/023309.html
  
this embeds the location of the libs into the python executable, avoiding the need
to manage LD_LIBRARY_PATH henceforth by doing the library search at link time rather 
than at run time
  
When building must:

    "set the environment variable LD_RUN_PATH to be the
     directory where the library will eventually be installed "

  
environment control
~~~~~~~~~~~~~~~~~~~~

without such a setup need to do fiddly management of 
the environment of the root::
    
  sudo bash -c ". $ENV_HOME/env.bash ; python- ; python " 
  
making env.bash executable and putting ENV_HOME into the path can do::

  sudo bash -c ". env.bash ; python- ; python " 
  
ldconfig
~~~~~~~~~

Systemwide, so mostly unwise for multi-use machines
  
  
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

    python-   

    local msg="=== $FUNCNAME :" 
    local nam=$(python-name)          ## MULTIPLE PYTHON FLEXIBILITY NOT WORTH THE HASSLE
    local tgz=$nam.tgz
    local ver=${nam/*-/}
    local url=http://www.python.org/ftp/python/$ver/$tgz

    [ -z "$nam" ] && echo $msg ABORT python-name is not defined .... maybe first do \"python- source\" && sleep 100000000000000000 && return 1

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
     ./configure --prefix=$(pythonbuild-prefix) --enable-shared 
}

pythonbuild-install(){

    cd $(pythonbuild-dir)
    make
    make install
}


pythonbuild-wipe(){
    
    local dir=$SYSTEM_BASE/python
    [ ! -d $dir ] && return 0 
    cd $dir
    rm -rf build
}

pythonbuild-wipe-install(){

   local dir=$SYSTEM_BASE/python
   [ ! -d $dir ] && return 0 
   cd $dir
   [ "${PYTHON_NAME:0:6}" != "Python" ] && echo bad PYTHON_NAME cannot proceed && return 1
   rm -rf $PYTHON_NAME
}


pythonbuild-again(){

    pythonbuild-wipe
    pythonbuild-wipe-install
    
    pythonbuild-get
    pythonbuild-configure
    pythonbuild-install

    pythonbuild-solibfix

    pythonbuild-setuptools-get

}


pythonbuild-solibfix(){

    local iwd=$PWD
    cd $PYTHON_HOME/lib/python2.5/config
    
    [ ! -L libpython2.5.so ] && ln -s ../../libpython2.5.so .
    ls -l 
    cd $iwd

}


pythonbuild-extras(){

    pythonbuild-setuptools-get 
    easy_install pip
}




pythonbuild-setuptools-get(){

  [ "$PYTHON_HOME/bin" == $(dirname $(which python)) ] || ( echo your path to python is incorrect aborting && return  )
  cd $SYSTEM_BASE/python 
  test -f ez_setup.py || curl -O  http://peak.telecommunity.com/dist/ez_setup.py
  mkdir -p $(python-site)    ##  perhaps need to use the lowtech one ?? python-site-

  python  ez_setup.py

   ## this puts easy_install in the PYTHON_HOME/bin
   which easy_install
 
}


