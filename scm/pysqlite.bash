#
#  pysqlite-get
#  pysqlite-install
#  pysqlite-test
#
#


export PYSQLITE_NAME=pysqlite-2.3.3

pysqlite-x(){ scp $SCM_HOME/pysqlite.bash ${1:-$TARGET_TAG}:$SCM_BASE ; }
pysqlite-i(){ . $SCM_HOME/pysqlite.bash ; }

pysqlite-get(){
  
  nam=$PYSQLITE_NAME
  tgz=$nam.tar.gz
  url=http://initd.org/pub/software/pysqlite/releases/2.3/2.3.3/$tgz
  
  cd $LOCAL_BASE
  test -d pysqlite || ( $SUDO mkdir pysqlite && $SUDO chown $USER pysqlite )
  cd pysqlite 

  test -f $tgz || curl -o $tgz $url
  test -d build || mkdir build
  test -d build/$nam || tar -C build -zxvf $tgz 
}

pysqlite-install(){

   # http://initd.org/pub/software/pysqlite/doc/install-source.html
   # NB the default python in path is too old: 2.2

   nam=$PYSQLITE_NAME
   cd $LOCAL_BASE/pysqlite/build/$nam

   perl -pi -e "s|^(include_dirs=)(.*)$|\$1$SQLITE_HOME/include|g" setup.cfg
   perl -pi -e "s|^(library_dirs=)(.*)$|\$1$SQLITE_HOME/lib|g"     setup.cfg

  # doesnt work with this version of python 
  #   /usr/bin/python2.4 setup.py build

   $PYTHON_HOME/bin/python setup.py build
   $PYTHON_HOME/bin/python setup.py install 

   # installs as a site-package into the installed python at
   #      $PYTHON_HOME/lib/python2.5/site-packages/pysqlite2
   #
}

pysqlite-test(){

    # need to run the test from a directory other than the build directory
    #  otherwise it fails with
    #   ImportError: No module named _sqlite
    cd 
    $PYTHON_HOME/bin/python << EOP
from pysqlite2 import test
test.test()
EOP

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
