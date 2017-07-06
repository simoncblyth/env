# === func-gen- : ui/qt4 fgp ui/qt4.bash fgn qt4 fgh ui
qt4-src(){      echo ui/qt4.bash ; }
qt4-source(){   echo ${BASH_SOURCE:-$(env-home)/$(qt4-src)} ; }
qt4-vi(){       vi $(qt4-source) ; }
qt4-env(){    
    elocal- ; 
    export QMAKESPEC=$(qt4-qmakespec)
    case $NODE_TAG in 
      N) PATH=/usr/local/Trolltech/Qt-4.8.4/bin:$PATH ;;
    esac
}
qt4-usage(){ cat << EOU

QT4
===

Installs
---------


D
~~

June 9, 2017 

macports install, many "didnt change anything" warnings 

   ~/macports/qt4-mac-jun9-2017.log 

::

    001 Last login: Fri Jun  9 12:28:30 on ttys001
      2 simon:~ blyth$ sudo port install qt4-mac +debug
      3 Password:
      4 Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
      5 --->  Fetching archive for libiconv
    ...
    423 --->  Some of the ports you installed have notes:
    424   dbus has the following notes:
    425     ############################################################################
    426     # Startup items have been generated that will aid in
    427     # starting dbus with launchd. They are disabled
    428     # by default. Execute the following commands to start them,
    429     # and to cause them to launch at startup:
    430     #
    431     # sudo launchctl load -w
    432     /Library/LaunchDaemons/org.freedesktop.dbus-system.plist
    433     # launchctl load -w /Library/LaunchAgents/org.freedesktop.dbus-session.plist
    434     ############################################################################
    435 simon:~ blyth$

::

    simon:~ blyth$ port contents qt4-mac | grep .app/Contents/Info.plist
    Warning: port definitions are more than two weeks old, consider updating them by running 'port selfupdate'.
      /Applications/MacPorts/Qt4/Assistant.app/Contents/Info.plist
      /Applications/MacPorts/Qt4/Designer.app/Contents/Info.plist
      /Applications/MacPorts/Qt4/Linguist.app/Contents/Info.plist
      /Applications/MacPorts/Qt4/QMLViewer.app/Contents/Info.plist
      /Applications/MacPorts/Qt4/pixeltool.app/Contents/Info.plist
      /Applications/MacPorts/Qt4/qdbusviewer.app/Contents/Info.plist
      /Applications/MacPorts/Qt4/qhelpconverter.app/Contents/Info.plist
      /Applications/MacPorts/Qt4/qttracereplay.app/Contents/Info.plist





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
  * delay until have an SL version which supports a new enough Qt out of its distro




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


N build of 4.8.4
------------------

* https://bugreports.qt-project.org/browse/QTBUG-19565

::

    g++ -c -include .pch/release-shared/QtCore -pipe -pthread -I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include -O2 -fvisibility=hidden -fvisibility-inlines-hidden -Wall -W -D_REENTRANT -fPIC -DQT_SHARED -DQT_BUILD_CORE_LIB -DQT_NO_USING_NAMESPACE -DQT_NO_CAST_TO_ASCII -DQT_ASCII_CAST_WARNINGS -DQT3_SUPPORT -DQT_MOC_COMPAT -DQT_USE_QSTRINGBUILDER -DELF_INTERPRETER=\"/lib/ld-linux.so.2\" -DQLIBRARYINFO_EPOCROOT -DHB_EXPORT=Q_CORE_EXPORT -DQT_NO_DEBUG -DQT_HAVE_MMX -DQT_HAVE_3DNOW -DQT_HAVE_SSE -DQT_HAVE_MMXEXT -DQT_HAVE_SSE2 -DQT_HAVE_SSE3 -DQT_HAVE_SSSE3 -D_LARGEFILE64_SOURCE -D_LARGEFILE_SOURCE -I../../mkspecs/linux-g++ -I. -I../../include -I../../include/QtCore -I.rcc/release-shared -Iglobal -I../../tools/shared -I../3rdparty/harfbuzz/src -I../3rdparty/md5 -I../3rdparty/md4 -I.moc/release-shared -o .obj/release-shared/qmutex_unix.o thread/qmutex_unix.cpp
    /usr/include/linux/futex.h:96: error: 'u32' was not declared in this scope
    /usr/include/linux/futex.h:96: error: 'uaddr' was not declared in this scope
    /usr/include/linux/futex.h:96: error: expected primary-expression before 'int'
    /usr/include/linux/futex.h:96: error: 'u32' was not declared in this scope
    /usr/include/linux/futex.h:96: error: expected primary-expression before 'unsigned'
    /usr/include/linux/futex.h:97: error: 'u32' was not declared in this scope
    /usr/include/linux/futex.h:97: error: 'uaddr2' was not declared in this scope
    /usr/include/linux/futex.h:97: error: 'u32' was not declared in this scope
    /usr/include/linux/futex.h:97: error: 'u32' was not declared in this scope
    /usr/include/linux/futex.h:97: error: initializer expression list treated as compound expression
    /usr/include/linux/futex.h:100: error: 'u32' was not declared in this scope
    /usr/include/linux/futex.h:100: error: 'uaddr' was not declared in this scope
    /usr/include/linux/futex.h:100: error: expected primary-expression before 'struct'
    /usr/include/linux/futex.h:100: error: expected primary-expression before 'int'
    /usr/include/linux/futex.h:100: error: initializer expression list treated as compound expression
    gmake[1]: *** [.obj/release-shared/qmutex_unix.o] Error 1
    gmake[1]: Leaving directory `/data1/env/local/env/ui/qt-everywhere-opensource-src-4.8.4/src/corelib'
    gmake: *** [sub-corelib-make_default-ordered] Error 2
    [blyth@belle7 qt-everywhere-opensource-src-4.8.4]$ 

Kludged it as suggested in the bugreport::

    [blyth@belle7 qt-everywhere-opensource-src-4.8.4]$ find . -name qmutex_unix.cpp
    ./src/corelib/thread/qmutex_unix.cpp
    [blyth@belle7 qt-everywhere-opensource-src-4.8.4]$ vi src/corelib/thread/qmutex_unix.cpp

     58 # include <mach/task.h>
     59 #elif defined(Q_OS_LINUX)
     60 // SCB start kludge  https://bugreports.qt-project.org/browse/QTBUG-19565
     61 //# include <linux/futex.h>
     62 # define FUTEX_WAIT 0
     63 # define FUTEX_WAKE 1
     64 // SCB end kludge
     65 # include <sys/syscall.h>
     66 # include <unistd.h>
     67 # include <QtCore/qelapsedtimer.h>
     68 #endif



EOU
}
qt4-mate(){ mate $(qt4-dir) ; }
qt4-dir(){ echo $(local-base)/env/ui/$(qt4-name) ; }
qt4-cd(){  cd $(qt4-dir); }

#qt4-name(){ echo qt-everywhere-opensource-src-4.7.4 ;  }
#qt4-url(){ echo http://download.qt-project.org/archive/qt/4.7/$(qt4-name).tar.gz ;  }

qt4-name(){ echo qt-everywhere-opensource-src-4.8.4 ;  }
qt4-url(){ echo http://download.qt-project.org/archive/qt/4.8/4.8.4/$(qt4-name).tar.gz ;  }


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


# macports 

qt4-html(){ open /opt/local/libexec/qt4/share/doc/html/index.html ; }


