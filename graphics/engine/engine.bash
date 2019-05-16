# === func-gen- : graphics/engine/engine fgp graphics/engine/engine.bash fgn engine fgh graphics/engine
engine-src(){      echo graphics/engine/engine.bash ; }
engine-source(){   echo ${BASH_SOURCE:-$(env-home)/$(engine-src)} ; }
engine-vi(){       vi $(engine-source) ; }
engine-env(){      elocal- ; }
engine-usage(){ cat << EOU

Engines
=========

* http://blog.digitaltutors.com/unity-udk-cryengine-game-engine-choose/


* Unity
* Unreal Engine 4 
* Source 2


Valve : Source 2
-------------------

* http://blog.digitaltutors.com/valve-enters-game-engine-race-source-2/


Render Pipeline for Panda3D
------------------------------


* https://github.com/tobspr/RenderPipeline


Godot
-------

* https://docs.godotengine.org/en/3.1/about/faq.html#which-platforms-are-supported-by-godot
* As of Godot 3.0, glTF is supported.


Geometry Shader

* https://github.com/godotengine/godot/issues/10817

* https://github.com/godotengine/godot/pull/28237

Sorry, the effort is enormously appreciated and admired, but going this way is
the wrong decision for the project for the following reasons:

* Geometry shaders are being deprecated in modern API versions. They are inefficient and can be better replaced with Compute.
* Geometry shaders are not well supported in modern hardware. They remain there for compatibility but they are not very efficient.
* Godot uses OpenGL ES 3.0, which does not support them, so they won't work on mobile.
* There is not high demand for the feature.

For Godot 4.0, the plan is to create compute shaders that do a similar
functions, but merging this for 3.x branch does not make much sense if it's
going to be removed in 4.0 months later.






EOU
}
engine-dir(){ echo $(local-base)/env/graphics/engine/graphics/engine-engine ; }
engine-cd(){  cd $(engine-dir); }
engine-mate(){ mate $(engine-dir) ; }
engine-get(){
   local dir=$(dirname $(engine-dir)) &&  mkdir -p $dir && cd $dir

}
