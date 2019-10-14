# === func-gen- : vr/openvr/openvr fgp vr/openvr/openvr.bash fgn openvr fgh vr/openvr
openvr-src(){      echo vr/openvr/openvr.bash ; }
openvr-source(){   echo ${BASH_SOURCE:-$(env-home)/$(openvr-src)} ; }
openvr-vi(){       vi $(openvr-source) ; }
openvr-env(){      elocal- ; }
openvr-usage(){ cat << EOU


OpenVR
========

* https://en.wikipedia.org/wiki/OpenVR

::

    OpenVR is a software development kit and application programming interface
    developed by Valve for supporting the SteamVR (HTC Vive)[1][2] and other
    virtual reality headset devices.[3][4] Valve has announced that they will be
    cooperating with the Open Source Virtual Reality (OSVR) project,[5] although
    the extent of the cooperation is unclear.[6]


* http://steamvr.com/
* http://steamcommunity.com/games/250820/announcements/detail/155715702499750866

* https://github.com/ValveSoftware/openvr
* https://github.com/ValveSoftware/openvr/wiki/API-Documentation
* https://github.com/ValveSoftware/openvr/tree/master/samples


* :google:`using openvr`

* http://www.roadtovr.com/making-valves-openvr-truly-inclusive-for-vr-headsets/
* http://jmonkeyengine.org/  3D engine




OpenVR is the API. SteamVR is the implementation of that API.
--------------------------------------------------------------

https://github.com/ValveSoftware/openvr/issues/291



Linux
-------

* https://www.gamingonlinux.com/articles/first-steps-with-openvr-and-the-vive-on-linux.7229
* http://steamcommunity.com/app/358040/discussions/0/351660338698372108/


Usage Example
---------------

* http://casual-effects.blogspot.tw/2016/03/opengl-sample-codeand-openvr-sample-code.html


Minimal Example
----------------

See ovrminimal-



EOU
}
openvr-dir(){ echo $(local-base)/env/vr/openvr ; }
openvr-cd(){  cd $(openvr-dir); }
openvr-get(){
   local dir=$(dirname $(openvr-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/ValveSoftware/openvr.git

}

openvr-libdir(){ echo $(openvr-dir)/lib/osx32 ; }





