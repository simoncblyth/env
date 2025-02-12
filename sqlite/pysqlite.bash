pysqlite-vi(){  vi $BASH_SOURCE ; }
pysqlite-usage(){ cat << EOU

pysqlite 
=========

* https://pypi.python.org/pypi/pysqlite
* https://code.google.com/p/pysqlite/

SQLite Python Binding Versions
----------------------------------

* https://code.google.com/p/pysqlite/downloads/list  

From June 2013 only the following pysqlite src distributions are available

* 2.6.3, 2.6.2, 2.6.0
* 2.5.6, 2.5.5

belle7 before and after static amalgamation install::

    [blyth@belle7 pysqlite-2.6.3]$ pysqlite-version
    ('/usr/lib/python2.4/site-packages/pysqlite2/dbapi2.pyc', '3.3.6', (3, 3, 6), '2.3.3', (2, 3, 3))

    [blyth@belle7 pysqlite-2.6.3]$ pysqlite-version 
    ('/usr/lib/python2.4/site-packages/pysqlite2/dbapi2.pyc', '3.7.17', (3, 7, 17), '2.6.3', (2, 6, 3))

NB the SQLite version 3.7.17 is distinct from the pysqlite one 2.6.3


Static amalgamation install
---------------------------

Old Redhat nodes use a python sqlite binding for their yum installations
which makes it problematic to upgrade to newer sqlite and python sqlite 
bindings using rpm OR yum techniques.

The older versions live under python module *sqlite*::

    [blyth@cms01 yum]$ /usr/bin/python -c "import sqlite as _ ; print (_.__file__,_._sqlite.sqlite_version(),_._sqlite.sqlite_version_info())"
    ('/usr/lib/python2.3/site-packages/sqlite/__init__.pyc', '3.3.6', (3, 3, 6))

To get around the impasse install pysqlite2 from its source distribution 
in *build_static* mode, in order to grab the lastest sqlite3 amalgamation and 
compile that directly into the python extension, avoiding issues of linking 
against the old shared sqlite libs, or accidentally upgrading them.

Build like that using the *pysqlite-* bash functions::

     cd ~/env
     svn up           # update env 
     env-
     pysqlite-

     pysqlite-get     # download the pysqlite source distribute

     pysqlite-get-amalgamation http://sqlite.org/2013/sqlite-amalgamation-3071700.zip

                      # determine the amalgamation url by perusing http://sqlite.org/download.html

     pysqlite-static-install

                      # installs into the system python at /usr/bin/python

     pysqlite-version 

                      # check the version is as expected for system python /usr/bin/python


The *pysqlite-version* function does::

    [blyth@cms01 yum]$ /usr/bin/python -c "from pysqlite2 import dbapi2 as _ ; print (_.__file__,_.sqlite_version,_.sqlite_version_info,_.version,_.version_info) "
    ('/usr/lib/python2.3/site-packages/pysqlite2/dbapi2.pyc', '3.7.17', (3, 7, 17), '2.6.3', (2, 6, 3))

Note that the updated version is typically accessed via python import::

     from pysqlite2 import dbapi2 as sqlite     

This means that existing users of the ancient python sqlite bindings such as yum are not disturbed, 
and can continue to::
 
     import sqlite



Build Static pysqlite
------------------------

It is static in the sense that it doesnt use the system sqlite shared lib, but 
rather uses the SQLite amalgamation statically compiled into the pysqlite2 extension.

::

    [blyth@cms01 pysqlite-2.6.3]$ /usr/bin/python setup.py build_static
    running build_static
    running build_py
    running build_ext
    building 'pysqlite2._sqlite' extension
    ...
    gcc -pthread -shared 
         build/temp.linux-i686-2.3/src/module.o 
         build/temp.linux-i686-2.3/src/connection.o 
         build/temp.linux-i686-2.3/src/cursor.o 
         build/temp.linux-i686-2.3/src/cache.o 
         build/temp.linux-i686-2.3/src/microprotocols.o 
         build/temp.linux-i686-2.3/src/prepare_protocol.o 
         build/temp.linux-i686-2.3/src/statement.o 
         build/temp.linux-i686-2.3/src/util.o 
         build/temp.linux-i686-2.3/src/row.o 
         build/temp.linux-i686-2.3/amalgamation/sqlite3.o
      -o build/lib.linux-i686-2.3/pysqlite2/_sqlite.so




Test Failures
----------------

grid1
      1/173 fails 
      
cms01
      cannot run due to lack of bz2, the 
      python build misses the module



EOU

}

pysqlite-env(){
    elocal-
}

pysqlite-name(){
   case ${1:-$NODE_TAG} in
    C2) echo pysqlite-2.6.3 ;;
     *) echo pysqlite-2.6.3 ;;
    GONE) echo pysqlite-2.3.3 ;;
   esac
}

pysqlite-home(){
   case $NODE_TAG in 
      H) echo $(local-base)/pysqlite/$(pysqlite-name) ;;
     XX) echo $(local-system-base)/pysqlite/$(pysqlite-name) ;;
      *) echo $(local-system-base)/pysqlite/$(pysqlite-name) ;;
   esac
}

pysqlite-builddir(){
    case $NODE_TAG in 
       H) echo $(local-base)/pysqlite/build/$(pysqlite-name) ;;
      XX) echo $(local-system-base)/pysqlite/build/$(pysqlite-name) ;;
       *) echo $(local-system-base)/pysqlite/build/$(pysqlite-name) ;;
    esac
}


pysqlite-again(){
   echo NOTE no wiping implemented ye  ... perhaps could be easy installed ? BUT do need a cfg change 
   pysqlite-get
   pysqlite-install
}

pysqlite-get(){
  local msg="=== $FUNCNAME :"
  local nam=$(pysqlite-name)
  local tgz=$nam.tar.gz
  local url=http://pysqlite.googlecode.com/files/$tgz

  echo $msg url $url
  local dir=$(dirname $(dirname $(pysqlite-builddir))) &&  mkdir -p $dir && cd $dir 
  [ ! -f "$tgz" ] && curl -L -O $url
  [ ! -d build/$nam ] && mkdir -p build && tar -C build -zxvf $tgz 
}

pysqlite-cd(){ cd $(pysqlite-builddir) ; }

pysqlite-get-amalgamation(){
  local msg="=== $FUNCNAME :"
  local url=$1
  local zip=$(basename $url)
  local nam=${zip/.zip}
  echo $msg url $url zip $zip nam $nam 

  pysqlite-cd
  [ ! -f "$zip" ] && curl -L -O $url
  mkdir -p amalgamation
  unzip -p $zip $nam/sqlite3.c > amalgamation/sqlite3.c
  unzip -p $zip $nam/sqlite3.h > amalgamation/sqlite3.h
}

pysqlite-wipe(){
  cd /usr/lib/python2.3/site-packages && sudo rm -rf pysqlite2 
}

pysqlite-static-install(){
  pysqlite-cd
  sudo /usr/bin/python setup.py build_static install
}

pysqlite-version(){
  /usr/bin/python -c "from pysqlite2 import dbapi2 as _ ; print (_.__file__,_.sqlite_version,_.sqlite_version_info,_.version,_.version_info) "
}








pysqlite-shared-install(){

 
   local msg="=== $FUNCNAME :"
   sqlite-
   [ -z $SQLITE_HOME ] && echo $msg ABORT no SQLITE_HOME && sleep 1000000
 
   cd $(pysqlite-builddir)

   echo $msg customizing setup.cfg with SQLITE_HOME : $SQLITE_HOME
   perl -pi -e "s|^(include_dirs=)(.*)$|\$1$SQLITE_HOME/include|g" setup.cfg
   perl -pi -e "s|^(library_dirs=)(.*)$|\$1$SQLITE_HOME/lib|g"     setup.cfg

   python-
   echo $msg $(which python)
   local cmd="$(python-sudo) python setup.py install "
   echo $msg $cmd 
   eval $cmd

   python-ls | grep sql

}


pysqlite-test(){

    # need to run the test from a directory other than the build directory
    #  otherwise it fails with
    #   ImportError: No module named _sqlite
    
    local iwd=$PWD
    cd /tmp
    
    python-
    echo $msg $(which python) ... LLP :
    echo $LD_LIBRARY_PATH | tr ":" "\n"
    python << EOP
from pysqlite2 import test
test.test()
EOP

    cd $iwd


   #
   # before setting LD_LIBRARY_PATH to include $SQLITE_HOME/lib  
   #  ... the import failed due to not finding  libsqlite3.so
   # 
   # apparently can avoid  having to set the path by changes to
   #   /etc/ld.so.conf and running ldconfig
   #         http://lists.initd.org/pipermail/pysqlite/2005-August/000125.html
   #
   #  get one failure of the tests:
   #
   # ======================================================================
   # FAIL: CheckSqlTimestamp (pysqlite2.test.types.DateTimeTests)
   # ----------------------------------------------------------------------
   # Traceback (most recent call last):
   #  File "/disk/d4/dayabay/local/python/Python-2.5.1/lib/python2.5/site-packages/pysqlite2/test/types.py", line 339, in CheckSqlTimestamp
   #	     self.failUnlessEqual(type(ts), datetime.datetime)
   #		 AssertionError: <type 'NoneType'> != <type 'datetime.datetime'>
   #		 
   #
   #
   #
   #[blyth@hfag pysqlite-2.3.3]$ pysqlite-test
   #Traceback (most recent call last):
   #  File "<stdin>", line 1, in <module>
   #	   File "pysqlite2/test/__init__.py", line 25, in <module>
   #	       from pysqlite2.test import dbapi, types, userfunctions, factory,
   #		   transactions,\
   #		     File "pysqlite2/test/dbapi.py", line 26, in <module>
   #			     import pysqlite2.dbapi2 as sqlite
   #				   File "pysqlite2/dbapi2.py", line 27, in <module>
   #				       from pysqlite2._sqlite import *
   #					   ImportError: No module named _sqlite
   #
   #    FIXED :  just run the test from outside the build directory ...
   #      http://lists.initd.org/pipermail/pysqlite/2006-January/000363.html
   #
   # 
}
