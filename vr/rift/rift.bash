# === func-gen- : vr/rift/rift fgp vr/rift/rift.bash fgn rift fgh vr/rift
rift-src(){      echo vr/rift/rift.bash ; }
rift-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rift-src)} ; }
rift-vi(){       vi $(rift-source) ; }
rift-env(){      elocal- ; }
rift-usage(){ cat << EOU

Oculus Rift
============

* https://developer.oculus.com/documentation/
* https://developer.oculus.com/documentation/pcsdk/latest/concepts/dg-sdk-setup/#dg_sdk_setup_installation
* https://forums.oculus.com

* http://oculusdrifter.blogspot.tw/2014/03/a-quick-guide-for-new-developers.html


Windows
--------

Solutions and project files for Visual Studio 2010, 2012 and 2013 are provided
with the SDK. Samples/Projects/Windows/VSxxxx/Samples.sln, or the 2012/2013
equivalent, is the main solution that allows you to build and run the samples,
and LibOVR itself.

SDK Overview
-------------

* http://static.oculus.com/documentation/pdfs/pcsdk/latest/dg.pdf
* https://developer.oculus.com/documentation/pcsdk/latest/concepts/dg-libovr/


* https://developer.oculus.com/documentation/pcsdk/latest/concepts/dg-render/#dg_render_distortion

The two virtual cameras in the scene should be positioned so that they are
pointing in the same direction (determined by the orientation of the HMD in the
real world), and such that the distance between them is the same as the
distance between the eyes, or interpupillary distance (IPD). This is typically
done by adding the ovrEyeRenderDesc::HmdToEyeViewOffset translation vector to
the translation component of the view matrix.

To target the Rift, you render the scene into one or two render textures,
passing these textures into the API. The Oculus runtime handles distortion
rendering, GPU synchronization, frame timing, and frame presentation to the
HMD.

::

   Presumably this means the SDK is applying the distortion appropriate to the lenses. 
   But what are camera param requirements ? 




EOU
}
rift-dir(){ echo $(local-base)/env/vr/rift/vr/rift-rift ; }
rift-cd(){  cd $(rift-dir); }
rift-mate(){ mate $(rift-dir) ; }
rift-get(){
   local dir=$(dirname $(rift-dir)) &&  mkdir -p $dir && cd $dir

}
