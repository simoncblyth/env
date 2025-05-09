modpython-src(){  echo apache/apachebuild/modpython.bash ; }
modpython-source(){ echo $(env-home)/$(modpython-src) ; }
modpython-vi(){      vi $(modpython-source) ; }

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
             ASUDO          : $ASUDO
    
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
        
        modpython-ldd
             if this does not have a python line the python is statically linked in
             ... check pythonbuild-solibfix and http://code.google.com/p/modwsgi/wiki/InstallationIssues
        
        





Now don't forget to edit your main config and add
    LoadModule python_module /data/env/system/apache/httpd-2.0.64/modules/mod_python.so
and if your configuration uses ClearModuleList, then also
    AddModule mod_python.c






        
     do it all again from scratch    
        $(type modpython-again)



 

EOU

}

modpython-apacheconf(){
     local msg="=== $FUNCNAME :"
     local conf=$(apache-conf)
     local tmp=/tmp/env/$FUNCNAME/$(basename $conf) && mkdir -p $(dirname $tmp)

     grep -q "^LoadModule python_module" $conf && echo $msg already done ... skipping && return  0

     cp $conf $tmp
     perl -pi -e 's,(# LoadModule foo_module modules/mod_foo.so),$1\nLoadModule python_module      modules/mod_python.so, ' $tmp
     echo $msg adding LoadModule line for modpython to $conf 
     diff $conf $tmp
     $ASUDO cp $tmp $conf
}


modpython-notes(){

cat << EON


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
  #local url=http://apache.cdpa.nsysu.edu.tw/httpd/modpython/$tgz
  local url=http://archive.apache.org/dist/httpd/modpython/$tgz


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


modpython-cd(){
  cd $(modpython-dir)
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
    $ASUDO make install
}



modpython-wipe(){

  local dir=$SYSTEM_BASE/mod_python
  [ ! -d $dir ] && return 0
  cd $dir
  rm -rf build

}


modpython-wipe-install(){

  local dir=$SYSTEM_BASE/mod_python
  [ ! -d $dir ] && return 0
  cd $dir
  
  [ "${MODPYTHON_NAME:0:10}" != "mod_python" ] && echo bad name $MODPYTHON_NAME cannot proceed && return 1
  
  rm -rf $MODPYTHON_NAME

}


modpython-again(){

   local msg="=== $FUNCNAME :"

    modpython-wipe
    modpython-wipe-install
    
    modpython-get
    modpython-configure
    modpython-install

    modpython-ldd


    echo $msg without this additon to the apachectl LD_LIBRARY_PATH get  libpython2.5.so not found when try to apachectl     
    apacheconf-
    apacheconf-envvars-add $PYTHON_HOME/lib
   

    modpython-apacheconf
 
}


modpython-ldd(){
    ldd $(apache-modulesdir)/mod_python.so
}





