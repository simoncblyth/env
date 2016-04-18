# === func-gen- : vr/openvr/ovrminimal fgp vr/openvr/ovrminimal.bash fgn ovrminimal fgh vr/openvr
ovrminimal-src(){      echo vr/ovrminimal/ovrminimal.bash ; }
ovrminimal-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ovrminimal-src)} ; }
ovrminimal-vi(){       vi $(ovrminimal-source) ; }
ovrminimal-env(){      elocal- ; }
ovrminimal-usage(){ cat << EOU

OpenGL OpenVR Minimal Example
===============================

http://casual-effects.blogspot.tw/2016/03/opengl-sample-codeand-openvr-sample-code.html


GLFW for window creation and events
GLEW for OpenGL extension loading
OpenVR for HMD initialization and tracking events, plus SteamVR's runtime

If you're working with G3D, those are already installed on your system. If not,
then you need to download the headers and binaries or source. No "installation"
of these libraries is needed, except for running SteamVR under Steam if you
wish to render in VR.




Without _VR definition get a simple interactive OpenGL window with cube geometry.
With _VR definition and without any HMD hooked up::

    simon:minimalOpenGL blyth$ ovrminimal-run
    Minimal OpenGL 4.1 Example by Morgan McGuire

    W, A, S, D, C, Z keys to translate
    Mouse click and drag to rotate
    ESC to quit

    OpenVR Initialization Error: Installation path could not be located (110)
    Assertion failed: (hmd), function main, file /usr/local/env/vr/minimalOpenGL/main.cpp, line 87.
    Abort trap: 6
    simon:minimalOpenGL blyth$ 


* https://github.com/ValveSoftware/openvr/wiki/HmdError

HmdError_Init_PathRegistryNotFound (110) - 
    The VR path registry file could not be read. 
    Reinstall the OpenVR runtime (or the SteamVR application on Steam.)

* https://www.reddit.com/r/SteamVR/comments/3zbqgr/trying_to_run_the_open_vr_sample_application/

* https://steamcommunity.com/app/358720/discussions/0/485624149150957321/

* https://developer.valvesoftware.com/wiki/SteamVR/steamvr.vrsettings

* http://media.steampowered.com/apps/steamvr/vr_setup.pdf

* https://docs.unrealengine.com/latest/INT/Platforms/SteamVR/QuickStart/1/index.html


EOU
}
ovrminimal-dir(){ echo $(local-base)/env/vr/minimalOpenGL ; }
ovrminimal-edir(){ echo $(env-home)/vr/ovrminimal ; }
ovrminimal-cd(){  cd $(ovrminimal-dir); }
ovrminimal-ecd(){  cd $(ovrminimal-edir); }
ovrminimal-mate(){ mate $(ovrminimal-dir) ; }
ovrminimal-get(){
   local dir=$(dirname $(ovrminimal-dir)) &&  mkdir -p $dir && cd $dir

   svn co svn://g3d.cs.williams.edu/g3d/G3D10/samples/minimalOpenGL
}

ovrminimal-init(){

   ovrminimal-cd

   local msg="$FUNCNAME : "
   local edir=$(ovrminimal-edir)
   local txt=CMakeLists.txt
   local etxt=$edir/$txt

   if [ -f "$txt" -a -f "$etxt" ]; then

      [ "$txt" -nt "$etxt" ] && echo $msg WARNING txt $txt newer than supposed source etxt $etxt && return 
   fi

   [ ! -f $etxt ] && echo $msg CANNOT FIND $etxt && return 
   [ ! -f $txt ] && echo $msg copying source etxt $etxt to txt $txt && cp $etxt . 

   [ -f "$txt" -a -f "$etxt" -a "$etxt" -nt "$txt" ] && echo $msg updating txt $txt from etxt $etxt && cp $etxt $txt

   echo $msg done 

}



ovrminimal-sdir(){ echo $(ovrminimal-dir) ; }
ovrminimal-bdir(){ echo $(local-base)/env/vr/minimalOpenGL.build ; }
ovrminimal-idir(){ echo $(local-base)/env/vr/minimalOpenGL.install ; }

ovrminimal-scd(){  cd $(ovrminimal-sdir); }
ovrminimal-bcd(){  cd $(ovrminimal-bdir); }
ovrminimal-icd(){  cd $(ovrminimal-idir); }

ovrminimal-bindir(){ echo $(ovrminimal-idir)/bin ; } 
ovrminimal-bin(){    echo $(ovrminimal-bindir)/$1 ; } 

ovrminimal-wipe(){
   local bdir=$(ovrminimal-bdir)
   rm -rf $bdir
}

ovrminimal-cmake(){
   local iwd=$PWD

   local bdir=$(ovrminimal-bdir)
   mkdir -p $bdir

   ovrminimal-bcd
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(ovrminimal-idir) \
       $(ovrminimal-sdir)

   cd $iwd
}

ovrminimal-make(){
   local iwd=$PWD

   ovrminimal-bcd
   make $*

   cd $iwd
}

ovrminimal-install(){
   ovrminimal-make install
}

ovrminimal--()
{
   ovrminimal-cmake
   ovrminimal-make
   ovrminimal-install
}

ovrminimal-run()
{
    local bin=$(ovrminimal-idir)/bin/OVRMINIMAL

    ovrminimal-scd   ## must run from dir with the shaders

    openvr-
    DYLD_LIBRARY_PATH=$(openvr-libdir) $bin

    ## huh why is RPATH setup not working ?
}

