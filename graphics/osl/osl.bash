# === func-gen- : graphics/osl/osl fgp graphics/osl/osl.bash fgn osl fgh graphics/osl src base/func.bash
osl-source(){   echo ${BASH_SOURCE} ; }
osl-edir(){ echo $(dirname $(osl-source)) ; }
osl-ecd(){  cd $(osl-edir); }
osl-dir(){  echo $LOCAL_BASE/env/graphics/osl/osl ; }
osl-cd(){   cd $(osl-dir); }
osl-vi(){   vi $(osl-source) ; }
osl-env(){  elocal- ; }
osl-usage(){ cat << EOU

Open Shading Language
========================

* https://en.wikipedia.org/wiki/Open_Shading_Language

Open Shading Language (OSL) is a shading language developed by Sony Pictures
Imageworks for use in its Arnold Renderer. It is also supported by Illumination
Research's 3Delight renderer,[1] Otoy's Octane Render,[2] V-Ray 3,[3] and by
the Cycles render engine in Blender (starting with Blender 2.65).[4] OSL's
surface and volume shaders define how surfaces or volumes scatter light in a
way that allows for importance sampling; thus, it is well suited for
physically-based renderers that support ray tracing and global illumination.


* http://opensource.imageworks.com/?p=osl
* http://opensource.imageworks.com/osl.html
* https://github.com/imageworks/OpenShadingLanguage/

OSL and OptiX
----------------

* https://github.com/imageworks/OpenShadingLanguage/issues/1170

18 May 2020:: 

    Optix 7 support has not been merged into master yet. 
    See #1111 for the current status (we hope to finish this soon).

    Also keep in mind that Optix support is highly experimental and is only
    relevant if you are planning on working on integrating OSL and Optix into your
    own renderer.


* https://github.com/imageworks/OpenShadingLanguage/pull/1111 "Adding support for OptiX 7"


Thoughts:

* this builds against OptiX 6 or 7 : very interesting for Opticks   


* :google:`NVIDIA OptiX OSL`





EOU
}
osl-get(){
   local dir=$(dirname $(osl-dir)) &&  mkdir -p $dir && cd $dir

}
