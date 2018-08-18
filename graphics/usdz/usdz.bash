# === func-gen- : graphics/usdz/usdz fgp graphics/usdz/usdz.bash fgn usdz fgh graphics/usdz
usdz-src(){      echo graphics/usdz/usdz.bash ; }
usdz-source(){   echo ${BASH_SOURCE:-$(env-home)/$(usdz-src)} ; }
usdz-vi(){       vi $(usdz-source) ; }
usdz-env(){      elocal- ; }
usdz-usage(){ cat << EOU

USD : Universal Scene Description
===================================

Apple ARKit 2, to support USDZ on macOS and iOS 

* https://appleinsider.com/articles/18/06/04/apples-arkit-2-and-usdz-usher-in-a-new-era-of-collaborative-augmented-reality
* https://theblog.adobe.com/introducing-project-aero/

* https://graphics.pixar.com/usd/files/USDZFileFormatSpecification.pdf


USDZ glTF
----------

* https://github.com/timvanscherpenzeel/gltf-to-usdz
* https://github.com/mrdoob/three.js/issues/14219

  Add support for USD and USDZ formats 

USD
-----

* http://openusd.org/
* https://graphics.pixar.com/usd/docs/api/index.html
* https://graphics.pixar.com/usd/docs/Introduction-to-USD.html
* https://graphics.pixar.com/usd/docs/Usdz-File-Format-Specification.html

* https://github.com/PixarAnimationStudios/USD
* https://github.com/PixarAnimationStudios/OpenSubdiv
* http://graphics.pixar.com/opensubdiv/docs/intro.html

::

    Hydra is the high-scalability, multi-pass, OpenSubdiv-based rendering
    architecture that ships as part of the USD distribution.  Its first and primary
    "back end" is a modern deferred-draw OpenGL implementation with support for
    pre-packaged and programmable glsl shaders, but it is intended to support
    multiple back-ends, just as it already supports multiple front-end clients
    simultaneously, and Pixar has already prototyped an OptiX-based backend for
    path-tracing.



EOU
}
usdz-dir(){ echo $(local-base)/env/graphics/usdz/graphics/usdz-usdz ; }
usdz-cd(){  cd $(usdz-dir); }
usdz-mate(){ mate $(usdz-dir) ; }
usdz-get(){
   local dir=$(dirname $(usdz-dir)) &&  mkdir -p $dir && cd $dir

}
