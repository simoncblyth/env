# === func-gen- : graphics/unreal/unreal fgp graphics/unreal/unreal.bash fgn unreal fgh graphics/unreal
unreal-src(){      echo graphics/unreal/unreal.bash ; }
unreal-source(){   echo ${BASH_SOURCE:-$(env-home)/$(unreal-src)} ; }
unreal-vi(){       vi $(unreal-source) ; }
unreal-env(){      elocal- ; }
unreal-usage(){ cat << EOU

Unreal Engine
===============

* https://www.unrealengine.com

Downloads "Epic Games Launcher.app"

* https://docs.unrealengine.com/latest/INT/
* https://docs.unrealengine.com/latest/INT/Platforms/SteamVR/index.html

* https://answers.unrealengine.com/index.html


glTF import from 4.19
------------------------

* https://forums.unrealengine.com/development-discussion/content-creation/1415582-the-new-gltf-import-supported-in-4-19-preview-is-awesome-experimentation


Tutorial
---------

* https://www.raywenderlich.com/771-unreal-engine-4-tutorial-for-beginners-getting-started


Linux
--------

* https://wiki.unrealengine.com/Building_On_Linux
* https://wiki.unrealengine.com/index.php?title=Building_On_Centos
* https://www.unixmen.com/linux-basics-install-centos-7-machine/

History
----------

* https://en.wikipedia.org/wiki/Unreal_Engine


In March 2014 Epic announced that the Unreal Engine 4 would no longer be
supporting UnrealScript, but instead support game scripting in C++. Visual
scripting would be supported by the Blueprints Visual Scripting system, a
replacement for the earlier Kismet visual scripting system.

As of March 2, 2015, Unreal Engine 4 is available to everyone for free, and all
future updates will be free,[95][96] with a selective royalty schedule


Associate Epic Account with Github Account
-------------------------------------------

* https://www.unrealengine.com/dashboard
* https://www.unrealengine.com/dashboard/settings

Then join the "Epic Games" group on github
to see the repositories.

* https://github.com/EpicGames
* https://github.com/EpicGames/UnrealEngine


Unreal CMake ?
----------------

* https://letstryunreal.wordpress.com

Unreal Geometry Shaders
------------------------

* https://forums.unrealengine.com/showthread.php?74531-Materials-amp-Geometry-shaders
* https://docs.unrealengine.com/latest/INT/Programming/Rendering/ShaderDevelopment/
* https://answers.unrealengine.com/questions/380379/geometry-shaders.html

Tools
--------

* https://github.com/baldurk/renderdoc

Tips
-----

* http://www.evermotion.org/tutorials/show/9714/seven-pro-tips-that-will-improve-your-unreal-visualizations



EOU
}
unreal-dir(){ echo $(local-base)/env/graphics/unreal/graphics/unreal-unreal ; }
unreal-cd(){  cd $(unreal-dir); }
unreal-mate(){ mate $(unreal-dir) ; }
unreal-get(){
   local dir=$(dirname $(unreal-dir)) &&  mkdir -p $dir && cd $dir

}
