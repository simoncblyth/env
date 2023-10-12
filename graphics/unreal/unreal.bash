# === func-gen- : graphics/unreal/unreal fgp graphics/unreal/unreal.bash fgn unreal fgh graphics/unreal
unreal-src(){      echo graphics/unreal/unreal.bash ; }
unreal-source(){   echo ${BASH_SOURCE:-$(env-home)/$(unreal-src)} ; }
unreal-vi(){       vi $(unreal-source) ; }
unreal-env(){      elocal- ; }
unreal-usage(){ cat << EOU

Unreal Engine
===============



Nanite Guide PDF
-----------------

* https://www.unrealengine.com/en-US/blog/download-the-new-nanite-teacher-s-guide

Now a core feature of Unreal Engine 5, Nanite is a new method for rendering 3D
graphics that intelligently renders only the detail that a viewer can perceive.
That means artists can now render extremely complex geometry in real-time,
without any performance limitations.

* ~/opticks_refs/nanite-for-educators-and-students-2-b01ced77f058.pdf 


Unreal Official Docs
----------------------

* https://docs.unrealengine.com/5.3/en-US/content-examples-sample-project-for-unreal-engine/
* https://docs.unrealengine.com/5.3/en-US/API/


Unreal Nanite on macOS : experimental
---------------------------------------

* https://forums.unrealengine.com/t/lumen-nanite-on-macos/508411/92

Github Unreal
---------------

* https://github.com/search?q=ue5&type=repositories

* https://github.com/Harrison1/unrealcpp
* https://unrealcpp.com/


* https://github.com/MonsterGuo/UE5_NvidiaAnsel

* https://github.com/philipturner/ue5-nanite-macos



UE5.2 : May 11, 2023
----------------------

* https://www.unrealengine.com/en-US/blog/unreal-engine-5-2-is-now-available


* https://9to5mac.com/2023/05/11/epic-unreal-enginenative-apple-silicon-mac/

Unreal Engine 5.2, which comes with native support for Apple Silicon Macs for
the first time.




History
--------

* https://en.wikipedia.org/wiki/Unreal_Engine

Unreal Engine 5 : formally launched for developers on April 5, 2022

One of its major features is Nanite, an engine that allows for high-detailed
photographic source material to be imported into games

Nanite can import nearly any other pre-existing three-dimension representation
of objects and environments, including ZBrush and CAD models, enabling the use
of film-quality assets.[118] Nanite automatically handles the levels of detail
(LODs) of these imported objects appropriate to the target platform and draw
distance, a task that an artist would have had to perform otherwise.[119]



Metal Developer Tools for Windows
-----------------------------------

* https://docs.unrealengine.com/4.26/en-US/SharingAndReleasing/Mobile/iOS/WindowsMetalShader/

Unreal Engine 4.26 onward can compile shaders for Apple's Metal API on a
Windows machine, greatly simplifying the workflow for iOS applications. To
enable this functionality, you need to install Apple's Metal Developer Tools
for Windows. Unreal Engine will automatically use this toolset once it is set
up. 



Intro
------


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
