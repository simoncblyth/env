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

* ~/opticks_refs/gltfOverview-2.0.0a.png

* https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md

* https://www.khronos.org/files/gltf20-reference-guide.pdf

* https://docs.microsoft.com/en-us/windows/mixed-reality/creating-3d-models-for-use-in-the-windows-mixed-reality-home


Double Sided
--------------

* https://github.com/KhronosGroup/glTF-Blender-Exporter/issues/58

Correct, but it is not a bug.
It has been decided - in glTF 2.0 - that double sided is a material and not a mesh attribute:
https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/schema/material.schema.json
If you need double sided, please use the the glTF 2.0 PBR node groups.

* https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/schema/material.schema.json


Looks like I need to update my oyocto ? Are missing doubleSided::

     644 struct material_t : glTFChildOfRootProperty_t {
     645     /// The emissive color of the material.
     646     std::array<float, 3> emissiveFactor = {{0, 0, 0}};
     647     /// The emissive map texture.
     648     textureInfo_t emissiveTexture = {};
     649     /// The normal map texture.
     650     material_normalTextureInfo_t normalTexture = {};
     651     /// The occlusion map texture.
     652     material_occlusionTextureInfo_t occlusionTexture = {};
     653     /// A set of parameter values that are used to define the metallic-roughness
     654     /// material model from Physically-Based Rendering (PBR) methodology.
     655     material_pbrMetallicRoughness_t pbrMetallicRoughness = {};
     656 };


Yep, latest yoctogl has it : and looks very changed, and simpler from before.

* https://github.com/xelatihy/yocto-gl/blob/master/yocto/yocto_gltf.h



Spec
-----

* https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md#nodes-and-hierarchy
  
  nodes can have an optional name property


names, extensions, extras
---------------------------

::

     218 ///
     219 /// Extensions
     220 ///
     221 using extension_t = std::map<std::string, json>;
     222 
     223 ///
     224 /// Extras
     225 ///
     226 using extras_t = json;
     227 
     228 // #codegen begin type ---------------------------------------------------------
     229 ///
     230 /// No description in schema.
     231 ///
     232 struct glTFProperty_t {
     233     /// No description in schema.
     234     //extension_t extensions = {};
     235     extension_t extensions{};
     236     /// No description in schema.
     237     extras_t extras = {};
     238 };
     239 
     240 ///
     241 /// No description in schema.
     242 ///
     243 struct glTFChildOfRootProperty_t : glTFProperty_t {
     244     /// The user-defined name of this object.
     245     std::string name = "";
     246 };



glTFChildOfRootProperty_t can have an optional name::

    epsilon:yocto blyth$ grep :\ glTFChildOfRootProperty_t yocto_gltf.h
    struct accessor_t : glTFChildOfRootProperty_t {
    struct animation_t : glTFChildOfRootProperty_t {
    struct buffer_t : glTFChildOfRootProperty_t {
    struct bufferView_t : glTFChildOfRootProperty_t {
    struct camera_t : glTFChildOfRootProperty_t {
    struct image_t : glTFChildOfRootProperty_t {
    struct texture_t : glTFChildOfRootProperty_t {
    struct material_t : glTFChildOfRootProperty_t {
    struct mesh_t : glTFChildOfRootProperty_t {
    struct node_t : glTFChildOfRootProperty_t {
    struct sampler_t : glTFChildOfRootProperty_t {
    struct scene_t : glTFChildOfRootProperty_t {
    struct skin_t : glTFChildOfRootProperty_t {


glTFProperty_t can have extras (unspecified json) and extensions (string keyed map of unspecified json).



extensions vs extras
~~~~~~~~~~~~~~~~~~~~~

* https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md#specifying-extensions
* https://github.com/KhronosGroup/glTF/blob/master/extensions/README.md

In addition to extensions, the extras object can also be used to extend glTF.
This is completely separate from extensions.  This enables glTF models to
contain application-specific properties without creating a full glTF extension.
This may be preferred for niche use cases where an extension would not be
widely adopted.

* extensions are formal, following similar pattern to OpenGL extensions (Vendor, EXT, etc)



Tools
-------

* https://github.com/magicien/GLTFSceneKit

  glTF loader for SceneKit, in Swift

* https://github.com/magicien/GLTFQuickLook
 
  macOS QuickLook plugin for glTF files



Tutorial
-----------

* https://github.com/KhronosGroup/glTF-Tutorials/blob/master/gltfTutorial/README.md

* https://github.com/KhronosGroup/glTF-Tutorials/blob/master/gltfTutorial/gltfTutorial_003_MinimalGltfFile.md

* https://github.com/KhronosGroup/glTF-Tutorials/blob/master/gltfTutorial/gltfTutorial_005_BuffersBufferViewsAccessors.md

* https://github.com/KhronosGroup/glTF-Tutorials/blob/master/gltfTutorial/gltfTutorial_010_Materials.md


Specification 1.0
-------------------

* https://github.com/KhronosGroup/glTF/tree/master/specification/1.0


* can include "extras" most anywhere for app specifics (eg CSG desc)


Specification 2.0 Draft?
----------------------------

* https://www.khronos.org/assets/uploads/developers/library/2017-gdc-webgl-webvr-gltf-meetup/7%20-%20glTF%20Update%20Feb17.pdf
* https://github.com/KhronosGroup/glTF/tree/2.0/specification/2.0

Schema Definition Using JSON Schema
-------------------------------------

* https://github.com/KhronosGroup/glTF/blob/2.0/specification/2.0/schema/glTF.schema.json

::

    {
        "$schema": "http://json-schema.org/draft-04/schema",
        "title": "glTF",
        "type": "object",
        "description": "The root object for a glTF asset.",
        "allOf": [ { "$ref": "glTFProperty.schema.json" } ],  # $ref is JSON pointer, analogous to XML XPath
        "properties": {
            ...
            "asset": {
                "allOf": [ { "$ref": "asset.schema.json" } ],
                "description": "Metadata about the glTF asset."
            },
            ...
            "meshes": {
                "type": "array",
                "description": "An array of meshes.",
                "items": {
                    "$ref": "mesh.schema.json"
                },
                "minItems": 1,
                "gltf_detailedDescription": "An array of meshes.  A mesh is a set of primitives to be rendered."
            },
            "nodes": {
                "type": "array",
                "description": "An array of nodes.",
                "items": {
                    "$ref": "node.schema.json"
                },
                "minItems": 1
            },
            ...
            "scene": {
                "allOf": [ { "$ref": "glTFid.schema.json" } ],
                "description": "The index of the default scene."
            },
            "scenes": {
                "type": "array",
                "description": "An array of scenes.",
                "items": {
                    "$ref": "scene.schema.json"
                },
                "minItems": 1
            },
            ...
            "extensions": { },
            "extras": { }
        },
        "dependencies": {
            "scene": [ "scenes" ]
        },
        "required": [ "asset" ]
    }


    ## glTFProperty.schema.json

    {
        "$schema": "http://json-schema.org/draft-04/schema",
        "title": "glTF property",
        "type": "object",
        "properties": {
            "extensions": {
                "$ref": "extension.schema.json"
            },
            "extras": {
                "$ref": "extras.schema.json"
            }
        }
    }

    ## extras.schema.json

    {
        "$schema": "http://json-schema.org/draft-04/schema",
        "title": "extras",
        "description": "Application-specific data."
    }

    ## means all the top level properties can have extras, what about mesh ?



::


allOf
~~~~~~~

To validate against allOf, the given data must be valid against all of the given subschemas.

* https://spacetelescope.github.io/understanding-json-schema/reference/combining.html#allof


json schema guides
~~~~~~~~~~~~~~~~~~~~~~

* http://json-schema.org
* https://spacetelescope.github.io/understanding-json-schema/
* https://spacetelescope.github.io/understanding-json-schema/reference/combining.html



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

gltf-png(){ open ~/opticks_refs/gltfOverview-2.0.0a.png ; }




