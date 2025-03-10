# === func-gen- : vr/vrworks/vrworks fgp vr/vrworks/vrworks.bash fgn vrworks fgh vr/vrworks
vrworks-src(){      echo vr/vrworks/vrworks.bash ; }
vrworks-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vrworks-src)} ; }
vrworks-vi(){       vi $(vrworks-source) ; }
vrworks-env(){      elocal- ; }
vrworks-usage(){ cat << EOU

VRWORKS : NVIDIAs SDK for VR
===============================

Pascal 
-------

* http://www.roadtovr.com/nvidia-explains-pascal-simultaneous-multi-projection-lens-matched-shading-for-vr/
* https://blogs.nvidia.com/blog/2016/05/06/pascal-vrworks/


New VRWorks-enabled branch of Unreal Engine Supports Pascal single pass stereo
---------------------------------------------------------------------------------

* https://developer.nvidia.com/vrworks-unreal-engine-branch-adds-support-vr-sli-lens-matched-shading-and-single-pass-stereo


VRWorks Audio SDK UE4 Plugin Using OptiX
-----------------------------------------------

* https://developer.nvidia.com/vrworks/vrworks-audio


VRWorks360 Video Stitching SDK : LightField Capture
------------------------------------------------------

* https://developer.nvidia.com/vrworks/vrworks-360video 


Unity VRWorks
---------------

Unity 2017.1.0 Beta 2 enables support for VRWorks 


* https://developer.nvidia.com/nvidia-vrworks-support-unity-engine-now-available

Download VRWorks-2.0-Package.zip
----------------------------------

* login as NVIDIA registered developer (NB not the ftp account)
* joined gameworks program, by filling in a form

Install::

   delta:env blyth$ mv ~/Downloads/VRWorks\ 2.0\ Package/ /usr/local/env/vr/VRWorks_2.0_Package





EOU
}
vrworks-dir(){ echo $(local-base)/env/vr/VRWorks_2.0_Package ; }
vrworks-cd(){  cd $(vrworks-dir); }

vrworks-sps-dir(){ echo $(vrworks-dir)/vr_multiprojection_ogl/gl_stereo_view_rendering ; }
vrworks-sps-cd(){ cd $(vrworks-sps-dir) ; }




