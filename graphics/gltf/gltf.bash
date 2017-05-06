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



Specification 1.0
-------------------

* https://github.com/KhronosGroup/glTF/tree/master/specification/1.0


* can include "extras" most anywhere for app specifics (eg CSG desc)


2.0 
-----

* https://www.khronos.org/assets/uploads/developers/library/2017-gdc-webgl-webvr-gltf-meetup/7%20-%20glTF%20Update%20Feb17.pdf
* https://github.com/KhronosGroup/glTF/tree/2.0/specification/2.0


glTF Samples
---------------

* https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/SimpleMeshes/glTF/SimpleMeshes.gltf

  Too simple 

* https://github.com/KhronosGroup/glTF-Sample-Models/blob/master/2.0/Lantern/glTF/Lantern.gltf

   


glTF Tutorials
----------------

Best source for detailed description.

* https://github.com/javagl/glTF-Tutorials/tree/master/gltfTutorial#gltf-tutorial



Scenes and Nodes
~~~~~~~~~~~~~~~~~~

* https://github.com/javagl/glTF-Tutorials/blob/master/gltfTutorial/gltfTutorial_004_ScenesNodes.md


Shaders included
~~~~~~~~~~~~~~~~~~~

* https://github.com/javagl/glTF-Tutorials/blob/master/gltfTutorial/gltfTutorial_012_ProgramsShaders.md


glTF Node Instancing
-----------------------

Gltf node instancing not supported in Cesium
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/AnalyticalGraphicsInc/cesium/issues/1754

 think that will do it. Essentially, in runtimeNode, we will transform the DAG
into a tree, then createRuntimeNodes will just work on a tree (no need to check
number of parents because it will always be one).


Node hierarchy - DAG or tree? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/KhronosGroup/glTF/issues/401


Example of node reuse as opposed to mesh reuse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/KhronosGroup/glTF/issues/276


Overview
---------

Uses approach very similar to the Opticks geocache, for the same reasons,
ie json and separate binary buffers.



gltf Loaders and Viewers
--------------------------

* https://github.com/KhronosGroup/glTF#c
* https://github.com/nvpro-pipeline/pipeline (Windows centric)

yocto-gl
~~~~~~~~~~~~

* https://github.com/xelatihy/yocto-gl  see yoctogl-

* https://github.com/xelatihy/yocto-gl/blob/master/yocto/yocto_gltf.h


Header only C++ Tiny glTF loader. (MIT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/syoyo/tinygltfloader

Simple OpenGL viewer for glTF geometry.

* https://github.com/syoyo/tinygltfloader/tree/master/examples/glview

Writing gltf

* https://github.com/syoyo/tinygltfloader/blob/master/examples/writer/writer.cc



laugh engine
~~~~~~~~~~~~~

A Vulkan implementation of real-time PBR renderer

* https://github.com/jian-ru/laugh_engine#laugh-engine
* http://jian-ru.github.io
* http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf


python-gltf-experiments (MIT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A sandbox repo for prototyping Python applications utilizing glTF.

* https://github.com/jzitelli/python-gltf-experiments

PyOpenGL
cyglfw3
PIL
NumPy (version 1.10 or later is required)
Pyrr
pyopenvr (optional, required for VR viewing)


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




EOU
}
gltf-dir(){ echo $(env-home)/graphics/gltf/tute ; }
gltf-minimal(){ echo $(gltf-dir)/minimal.gltf ; }
gltf-minimal-vi(){ vi -R $(gltf-minimal) ; }


gltf-cd(){  cd $(gltf-dir); }
gltf-get(){
   local dir=$(dirname $(gltf-dir)) &&  mkdir -p $dir && cd $dir

}



