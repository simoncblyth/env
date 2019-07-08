# === func-gen- : vr/steamvr/steamvr fgp vr/steamvr/steamvr.bash fgn steamvr fgh vr/steamvr
steamvr-src(){      echo vr/steamvr/steamvr.bash ; }
steamvr-source(){   echo ${BASH_SOURCE:-$(env-home)/$(steamvr-src)} ; }
steamvr-vi(){       vi $(steamvr-source) ; }
steamvr-env(){      elocal- ; }
steamvr-usage(){ cat << EOU

SteamVR
=========


On Linux
----------

* https://github.com/ValveSoftware/SteamVR-for-Linux/blob/master/README.md

* https://www.cgl.ucsf.edu/chimera/data/linux-vr-oct2018/linuxvr.html


* https://www.reddit.com/r/SteamVR/comments/aqd5gz/steamvr_and_linux_how_is_it/

* http://cheesetalks.net/proton-linux-gaming-history.php


Steam Runtime : chroot environment
--------------------------------------

* https://github.com/ValveSoftware/steam-runtime


Vive Setup
------------

* http://media.steampowered.com/apps/steamvr/vr_setup.pdf


Destinations
--------------

* https://developer.valvesoftware.com/wiki/Destinations

Source SDK
-----------

* https://developer.valvesoftware.com/wiki/Category:Source_SDK_FAQ



EOU
}
steamvr-dir(){ echo $(local-base)/env/vr/steamvr/vr/steamvr-steamvr ; }
steamvr-cd(){  cd $(steamvr-dir); }
steamvr-mate(){ mate $(steamvr-dir) ; }
steamvr-get(){
   local dir=$(dirname $(steamvr-dir)) &&  mkdir -p $dir && cd $dir

}
