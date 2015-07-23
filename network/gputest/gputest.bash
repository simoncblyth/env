# === func-gen- : network/gputest/gputest fgp network/gputest/gputest.bash fgn gputest fgh network/gputest
gputest-src(){      echo network/gputest/gputest.bash ; }
gputest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gputest-src)} ; }
gputest-vi(){       vi $(gputest-source) ; }
gputest-env(){      elocal- ; }
gputest-usage(){ cat << EOU

GPUTEST at IHEP
=================

Access
-------

* ssh lxslc509.ihep.ac.cn
* ssh gputest.ihep.ac.cn

Start TurboVNC::

   turbovnc-viewer

* in *Options > Security* set:
 
  * SSH user:blyth
  * SSH host:lxslc509.ihep.ac.cn
  * leave "Use VNC server as gateway" unchecked

* in main little window set eg:

  * VNC server:gputest.ihep.ac.cn:2
  * The port ":2" is listed on starting the vncserver on the target node

* NB TurboVNC operates with a single client per vncserver, 
  so need to to ssh in and start the vncserver first, and 
  use the port eg ":2" listed in the connection dialog above  

* to run OpenGL apps via VirtualGL prefix with "vglrun"


Start VNCServer
----------------

::
 
   /opt/TurboVNC/bin/vncserver
   Xorg





EOU
}
gputest-dir(){ echo $(local-base)/env/network/gputest/network/gputest-gputest ; }
gputest-cd(){  cd $(gputest-dir); }
gputest-mate(){ mate $(gputest-dir) ; }
gputest-get(){
   local dir=$(dirname $(gputest-dir)) &&  mkdir -p $dir && cd $dir

}
