# === func-gen- : ui/qt4 fgp ui/qt4.bash fgn qt4 fgh ui
qt4-src(){      echo ui/qt4.bash ; }
qt4-source(){   echo ${BASH_SOURCE:-$(env-home)/$(qt4-src)} ; }
qt4-vi(){       vi $(qt4-source) ; }
qt4-env(){    
    elocal- ; 
    export QMAKESPEC=$(qt4-qmakespec)
}
qt4-usage(){ cat << EOU

QT4
===

Installs
---------

N
~~


yum qt4 on belle7 is too old for meshlab build

qmake binary hidden on SL5 ? Have to adjust PATH::

    [blyth@belle7 ~]$ which qmake
    /usr/lib/qt4/bin/qmake

::

    [blyth@belle7 meshlab]$ rpm -ql qt4-devel | grep qmake
    /usr/lib/qt4/bin/qmake

Look into source install

* http://belle7.nuu.edu.tw/qt4/
* http://belle7.nuu.edu.tw/qt4/requirements-x11.html

  * so many dependencies, maybe not worth it : I just wanted to be able to profile 
    the collada loading and cannot install gperftools on PPC


G
~~

* http://localhost/qt4/

::

    simon:~ blyth$ qmake -v
    QMake version 2.01a
    Using Qt version 4.8.5 in /opt/local/lib


* macports qt4-mac-devel failed after many hours of building, 

  * http://trac.macports.org/ticket/34902
  * http://trac.macports.org/changeset/96486

* instead installed macports qt4-mac which literally took days gto build::

    simon:~ blyth$ sudo port -v install qt4-mac
    ...
    ./opt/local/share/qt4/translations/qvfb_ru.qm
    ./opt/local/share/qt4/translations/qvfb_sl.qm
    ./opt/local/share/qt4/translations/qvfb_uk.qm
    ./opt/local/share/qt4/translations/qvfb_zh_CN.qm
    ./opt/local/share/qt4/translations/qvfb_zh_TW.qm
    NOTE: Qt database plugins for mysql55, postgresql91, and sqlite2 are NOT installed by this port; they are installed by qt4-mac-*-plugin instead.
    --->  Cleaning qt4-mac
    --->  Removing work directory for qt4-mac
    --->  Updating database of binaries: 100.0%
    --->  Scanning binaries for linking errors: 100.0%
    --->  No broken files found.


macports debugging
^^^^^^^^^^^^^^^^^^^^
::

    simon:local blyth$ port dir qt4-mac-devel
    /opt/local/var/macports/sources/rsync.macports.org/release/ports/aqua/qt4-mac-devel
    simon:local blyth$ cd /opt/local/var/macports/sources/rsync.macports.org/release/ports/aqua/qt4-mac-devel



qmake : OLD MAC needs g++ not clang++
----------------------------------------

::

    simon:mkspecs blyth$ pwd
    /opt/local/share/qt4/mkspecs

    simon:mkspecs blyth$ find . -name '*.conf' -exec grep -H "QMAKE_CXX " {} \;
    ./common/clang.conf:QMAKE_CXX = clang++
    ./common/g++-base.conf:QMAKE_CXX = /usr/bin/g++-4.2
    ./common/llvm.conf:QMAKE_CXX = llvm-g++
    ./common/mac.conf:QMAKE_CXX = $(CCACHE) $$QMAKE_CXX
    ./freebsd-g++46/qmake.conf:QMAKE_CXX          = g++46
    ./linux-arm-gnueabi-g++/qmake.conf:QMAKE_CXX               = arm-linux-gnueabi-g++
    ./macx-g++40/qmake.conf:QMAKE_CXX        = g++-4.0
    ./macx-g++42/qmake.conf:QMAKE_CXX        = g++-4.2
    ./qnx-armv7le-qcc/qmake.conf:QMAKE_CXX               = qcc -Vgcc_ntoarmv7le
    ./qnx-x86-qcc/qmake.conf:QMAKE_CXX               = qcc -Vgcc_ntox86
    ./qws/integrity-x86-cx86/qmake.conf:QMAKE_CXX               = cxint86
    ./qws/linux-arm-g++/qmake.conf:QMAKE_CXX               = arm-linux-g++
    ./qws/linux-arm-gnueabi-g++/qmake.conf:QMAKE_CXX               = arm-none-linux-gnueabi-g++
    ./qws/linux-armv6-g++/qmake.conf:QMAKE_CXX               = arm-linux-g++
    ...

::

    simon:mkspecs blyth$ ll /usr/bin/g++*
    -rwxr-xr-x  1 root  wheel   93088  5 Feb  2009 /usr/bin/g++-4.0
    -rwxr-xr-x  1 root  wheel  105680  7 Jul  2009 /usr/bin/g++-4.2
    lrwxr-xr-x  1 root  wheel       7 28 Aug  2010 /usr/bin/g++ -> g++-4.0
    simon:mkspecs blyth$ 

qmake : fails to override CXX
--------------------------------

::

    simon:common blyth$ qmake QMAKE_CC=garbage QMAKE_CXX=cxxgarbage

    6 # Command: /opt/local/bin/qmake QMAKE_CC=garbage QMAKE_CXX=cxxgarbage -o Makefile common.pro
    ...
    11 CC            = garbage
    12 CXX           = clang++ 


macports ticket
----------------

* https://trac.macports.org/ticket/35982
* https://trac.macports.org/browser/trunk/dports/aqua/qt4-mac/Portfile   YUCK: ~1200 lines 



EOU
}
qt4-mate(){ mate $(qt4-dir) ; }
qt4-dir(){ echo $(local-base)/env/ui/$(qt4-name) ; }
qt4-cd(){  cd $(qt4-dir); }
qt4-name(){ echo qt-everywhere-opensource-src-4.7.4 ;  }
qt4-url(){ echo http://download.qt-project.org/archive/qt/4.7/$(qt4-name).tar.gz ;  }
qt4-get(){
   local dir=$(dirname $(qt4-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(qt4-url)
   local tgz=$(basename $url)
   [ ! -f "$tgz" ] && curl -L -O "$url"
   local nam=${tgz/.tar.gz}
   [ ! -d "$nam" ] && tar zxvf $tgz
}



qt4-qmakespec(){
  case $NODE_TAG in 
    G) echo /opt/local/share/qt4/mkspecs/macx-g++ ;; 
  esac
}
qt4-info(){
   echo QMAKESPEC $QMAKESPEC
}
qt4-kludge(){
  local cmd="find . -name Makefile -exec perl -pi -e 's,clang,g,g' {} \;"
  echo $cmd
  eval $cmd
  find . -name Makefile -exec grep -H "CXX " {} \;
}
qt4-specs(){
  cd /opt/local/share/qt4/mkspecs
}
qt4-docs(){
  apache-
  cd `apache-htdocs`
  sudo ln -sf /opt/local/share/doc/qt4/html qt4
}

