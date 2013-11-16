# === func-gen- : ui/qt4 fgp ui/qt4.bash fgn qt4 fgh ui
qt4-src(){      echo ui/qt4.bash ; }
qt4-source(){   echo ${BASH_SOURCE:-$(env-home)/$(qt4-src)} ; }
qt4-vi(){       vi $(qt4-source) ; }
qt4-env(){      elocal- ; }
qt4-usage(){ cat << EOU

QT4
===

Installs
---------

G
~~

::

    port install qt4-mac-devel 

Fails with::

    
       

* http://trac.macports.org/ticket/34902
* http://trac.macports.org/changeset/96486

::

    simon:local blyth$ port dir qt4-mac-devel
    /opt/local/var/macports/sources/rsync.macports.org/release/ports/aqua/qt4-mac-devel
    simon:local blyth$ cd /opt/local/var/macports/sources/rsync.macports.org/release/ports/aqua/qt4-mac-devel

Instead installed qt4-mac, a jumbo install::

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
    simon:~ blyth$ 
    simon:~ blyth$ 
    simon:~ blyth$ sudo port -v install qt4-mac


EOU
}
qt4-dir(){ echo $(local-base)/env/ui/ui-qt4 ; }
qt4-cd(){  cd $(qt4-dir); }
qt4-mate(){ mate $(qt4-dir) ; }
qt4-get(){
   local dir=$(dirname $(qt4-dir)) &&  mkdir -p $dir && cd $dir

}
