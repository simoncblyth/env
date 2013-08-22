# === func-gen- : graphics/vrml/freewrl fgp graphics/vrml/freewrl.bash fgn freewrl fgh graphics/vrml
freewrl-src(){      echo graphics/vrml/freewrl.bash ; }
freewrl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(freewrl-src)} ; }
freewrl-vi(){       vi $(freewrl-source) ; }
freewrl-env(){      elocal- ; }
freewrl-usage(){ cat << EOU

FREEWRL
========

OSX
----


http://freewrl.sourceforge.net/build_OSX.html

   * OSX build needs a more recent XCode than mine I guess.


Linux
-------

http://freewrl.sourceforge.net/install_Linux.html

Trying a Linux build on OSX gives::


    simon:freex3d blyth$ ./configure --with-statusbar=hud
    ...
    configure: Determining Javascript engine to build against
    checking for spidermonkey >= 1.7.0 while ... checking for JAVASCRIPT_ENGINE... no
    configure: error: Unable to find an appropriate javascript engine



EOU
}
freewrl-dir(){ echo $(local-base)/env/graphics/vrml/graphics/freewrl ; }
freewrl-cd(){  cd $(freewrl-dir); }
freewrl-mate(){ mate $(freewrl-dir) ; }
freewrl-get(){
   local dir=$(dirname $(freewrl-dir)) &&  mkdir -p $dir && cd $dir

   cvs -z3 -d:pserver:anonymous@freewrl.cvs.sourceforge.net:/cvsroot/freewrl checkout -P freewrl


}
