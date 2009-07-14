pysqlite-vi(){  vi $BASH_SOURCE ; }
pysqlite-usage(){

  cat << EOU

     pysqlite-name     :  $(pysqlite-name)
     pysqlite-home     :  $(pysqlite-home)
     pysqlite-builddir : $(pysqlite-builddir)
     
     $(type pysqlite-again)
    
           NOTE no wiping implemented ye  ... perhaps could be easy installed ? BUT do need a cfg change

     pysqlite-get

     pysqlite-install

     pysqlite-test
         1/173 fails on grid1
         cannot run on cms01 due to lack of bz2 
         (python build on cms01 missing the module)


EOU

}

pysqlite-env(){
    elocal-
}

pysqlite-name(){
    echo pysqlite-2.3.3
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
  
  local nam=$(pysqlite-name)
  local tgz=$nam.tar.gz
  local url=http://initd.org/pub/software/pysqlite/releases/2.3/2.3.3/$tgz
  
  cd $SYSTEM_BASE
  mkdir -p pysqlite
  cd pysqlite 

  test -f $tgz || curl -O $url
  mkdir -p  build
  test -d build/$nam || tar -C build -zxvf $tgz 
}



pysqlite-install(){

   # http://initd.org/pub/software/pysqlite/doc/install-source.html
 
   sqlite-
   [ -z $SQLITE_HOME ] && echo $msg ABORT no SQLITE_HOME && sleep 1000000
 
   cd $(pysqlite-builddir)

   perl -pi -e "s|^(include_dirs=)(.*)$|\$1$SQLITE_HOME/include|g" setup.cfg
   perl -pi -e "s|^(library_dirs=)(.*)$|\$1$SQLITE_HOME/lib|g"     setup.cfg

   python-
   python setup.py install 

}


pysqlite-test(){

    # need to run the test from a directory other than the build directory
    #  otherwise it fails with
    #   ImportError: No module named _sqlite
    
    local iwd=$PWD
    cd /tmp
    
    python-
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
