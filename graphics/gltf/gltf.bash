# === func-gen- : graphics/gltf/gltf fgp graphics/gltf/gltf.bash fgn gltf fgh graphics/gltf
gltf-src(){      echo graphics/gltf/gltf.bash ; }
gltf-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gltf-src)} ; }
gltf-vi(){       vi $(gltf-source) ; }
gltf-env(){      elocal- ; }
gltf-usage(){ cat << EOU

glTF
=====

GL Transmission Format (glTF) from The Khronos Group aims to provide a
lightweight, efficient format meant for 3d scene representation in a way that
could be easily streamed, e.g. over the internet.


* https://www.khronos.org/news/press/khronos-collada-now-recognized-as-iso-standard
* https://github.com/KhronosGroup/glTF/blob/master/specification/README.md
* https://www.khronos.org/gltf
* https://github.com/KhronosGroup/glTF#gltf-tools

glTF Tutorials
----------------

* https://github.com/javagl/glTF-Tutorials/tree/master/gltfTutorial#gltf-tutorial

Overview
---------

Uses approach very similar to the Opticks geocache, for the same reasons,
ie json and separate binary buffers.

Questions
-----------

NPY buffers within gltf ? Looks like YES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Can .npy buffers directly be used within gltf just by 
appropriate specification of buffer offets for the NPY headers ?

* https://github.com/javagl/glTF-Tutorials/blob/master/gltfTutorial/gltfTutorial_005_BuffersBufferViewsAccessors.md
* just needs header length method in NPY 

NPY Format
~~~~~~~~~~~~~

* https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html

The next HEADER_LEN bytes form the header data describing the array’s format.
It is an ASCII string which contains a Python literal expression of a
dictionary. It is terminated by a newline (‘n’) and padded with spaces (‘x20’)
to make the total length of the magic string + 4 + HEADER_LEN be evenly
divisible by 16 for alignment purposes.


Why gltf is interesting ...
------------------------------

* Can benefit from other peoples work on realistic physically based rendering,
  may allow Opticks geometries and event records to be visualizable 
  with a growing collection of viewers including VR ones.
  
* an emerging standard, many 3D tools and end user applications will be supporting it 


gltf Loaders and Viewers
--------------------------

* https://github.com/KhronosGroup/glTF#c

* https://github.com/nvpro-pipeline/pipeline (Windows centric)

yocto-gl
~~~~~~~~~~~~

* https://github.com/xelatihy/yocto-gl  see yoctogl-


laugh engine
~~~~~~~~~~~~~

A Vulkan implementation of real-time PBR renderer

* https://github.com/jian-ru/laugh_engine#laugh-engine
* http://jian-ru.github.io
* http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf






EOU
}
gltf-dir(){ echo $(local-base)/env/graphics/gltf/graphics/gltf-gltf ; }
gltf-cd(){  cd $(gltf-dir); }
gltf-mate(){ mate $(gltf-dir) ; }
gltf-get(){
   local dir=$(dirname $(gltf-dir)) &&  mkdir -p $dir && cd $dir

}
