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





EOU
}
qt4-dir(){ echo $(local-base)/env/ui/ui-qt4 ; }
qt4-cd(){  cd $(qt4-dir); }
qt4-mate(){ mate $(qt4-dir) ; }
qt4-get(){
   local dir=$(dirname $(qt4-dir)) &&  mkdir -p $dir && cd $dir

}
