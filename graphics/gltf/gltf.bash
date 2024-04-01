# === func-gen- : graphics/gltf/gltf fgp graphics/gltf/gltf.bash fgn gltf fgh graphics/gltf
gltf-src(){      echo graphics/gltf/gltf.bash ; }
gltf-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gltf-src)} ; }
gltf-vi(){       vi $(gltf-source) ; }
gltf-env(){      elocal- ; }
gltf-usage(){ cat << EOU

glTF
=====



gltf accessors componentType
------------------------------

* https://github.com/KhronosGroup/glTF-Tutorials/blob/main/gltfTutorial/gltfTutorial_005_BuffersBufferViewsAccessors.md

::

    5126 (FLOAT) 
    5123 (UNSIGNED_SHORT)



Impls
-------

* https://github.com/syoyo/tinygltf


Projects using tinygltf include:

* http://www.open3d.org/
* http://www.open3d.org/docs/release/



* https://github.com/google/usd_from_gltf




Intro 
-------

GL Transmission Format (glTF) from The Khronos Group aims to provide a
lightweight, efficient format meant for 3d scene representation in a way that
could be easily streamed, e.g. over the internet.

* https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md

  Best to read the spec : its the fullest description 



* https://www.khronos.org/news/press/khronos-collada-now-recognized-as-iso-standard
* https://github.com/KhronosGroup/glTF/blob/master/specification/README.md
* https://www.khronos.org/gltf
* https://github.com/KhronosGroup/glTF#gltf-tools

* ~/opticks_refs/gltfOverview-2.0.0a.png

* https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md

* https://www.khronos.org/files/gltf20-reference-guide.pdf

* https://docs.microsoft.com/en-us/windows/mixed-reality/creating-3d-models-for-use-in-the-windows-mixed-reality-home

glTF 2.0 PBR
-----------------

* https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md#metallic-roughness-material


* https://www.allegorithmic.com/pbr-guide
* https://academy.allegorithmic.com/courses/b6377358ad36c444f45e2deaa0626e65

  Substance Artists guide to PBR

  * F0 : Fresnel zero : reflectance at 0 degress (normal incidence)
  * F0 = (n-1)^2/(n+1)^2

* https://cesium.com/blog/2017/08/08/physically-based-rendering-in-cesium/


* https://github.com/KhronosGroup/glTF-WebGL-PBR

A surface using the Metallic-Roughness material is governed by three parameters:

* base color (albedo)
* metallicness
* roughness. 

Either the parameters can be given constant values, which would dictate the
shading of an entire mesh uniformly, or textures can be provided that map
varying values over a mesh. In this project, all of the glTF files followed the
latter case. It is important to note here that although metallic and roughness
are separate parameters, they are provided as a single texture in which the
metallic values are in the blue channel and the roughness values are in the
green channel to save on space

* http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf

  Real Shading in Unreal Engine 4
  by Brian Karis, Epic Games   (despite the title it is mostly general discussion of PBR)  


  



glTF viewer/tools on Windows ? Because NVIDIA ShadowPlay movie capture is on windows only
-------------------------------------------------------------------------------------------------

* https://github.com/KhronosGroup/glTF/blob/master/README.md#gltf-tools


glTF 2.0 Viewer written in Rust
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/bwasty/gltf-viewer/blob/master/CHANGELOG.md#031---2018-03-16

::

   epsilon:Downloads blyth$ ./gltf-viewer /tmp/X4SolidTest/cathode.gltf 
   [ERROR] glTF import failed: Validation([(Path("meshes[0].primitives[0].material"), IndexOutOfBounds)])
   epsilon:Downloads blyth$ 

Hmm, unlike other viewers this one doesnt accept material:"0" as a default material. Removing that 
line can see the cathode.

Same issue and fix with::

    ./gltf-viewer -v -v -v /tmp/X4MeshTest/X4MeshTest.gltf 

TODO: try with larger geometry 


Lugdunum, a modern 3D engine using the Vulkan API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/Lugdunum3D/Lugdunum
* https://lugdunum3d.github.io
* https://lugdunum3d.github.io/doc/build.html



Microsoft.glTF.CPP
~~~~~~~~~~~~~~~~~~~~

* https://www.nuget.org/packages/Microsoft.glTF.CPP/


Paint 3D support for GLB
~~~~~~~~~~~~~~~~~~~~~~~~~~

Microsoft has announced support for glTF in Paint 3D

* https://github.com/KhronosGroup/glTF/issues/1037
* https://blogs.windows.com/windowsexperience/2017/07/12/announcing-more-updates-to-paint-3d-magic-select-enhancements-drawing-tools/

Paint 3D also now supports a new industry-wide open standard for 3D file
sharing called GLB, a part of gLTF (GL Transmission Format). This allows for
faster and more efficient transfer of 3D assets by outputting only one
container for all assets, minimizing the file size and the ability to use the
files across other programs as a universal format.


Windows glTF toolkit
~~~~~~~~~~~~~~~~~~~~~~~

convert a glTF 2.0 core asset for use in the Windows Mixed Reality home

* https://github.com/microsoft/gltf-toolkit

* https://docs.microsoft.com/en-us/windows/mixed-reality/creating-3d-models-for-use-in-the-windows-mixed-reality-home


Windows Mixed Reality **home** (app launcher 3D environment) has restrictions on models:

* Assets must be delivered in the .glb file format (binary glTF)
* Assets must be less than 10k triangles, have no more than 64 nodes and 32 submeshes per LOD) 



PBR
-----

* https://www.khronos.org/assets/uploads/developers/library/2017-gtc/glTF-2.0-and-PBR-GTC_May17.pdf


Fabricated Top Node, switching Z<->Y 
-------------------------------------------

::

    343469     {
    343470       "name": "Fabricated top node 12230",
    343471       "children": [
    343472         3152
    343473       ],
    343474       "matrix": [
    343475         1.0, 0.0, 0.0, 0.0,
    343476         0.0, 0.0, 1.0, 0.0,
    343477         0.0, 1.0, 0.0, 0.0,
    343478         0.0, 0.0, 0.0, 1.0
    343479       ],
    343480     } 
    343481   ],
    343482   "scenes": [
    343483     {
    343484       "nodes": [
    343485         12230
    343486       ] 
    343487     } 
    343488   ] 




2018 Web3D Conference Keynote, Neil Trevett 
----------------------------------------------

* https://www.khronos.org/assets/uploads/developers/library/2018-web3d/Web3D-Keynote-Poznan-2D_Jun18.pdf


* https://www.anandtech.com/show/12894/apple-deprecates-opengl-across-all-oses

* https://moltengl.com/moltenvk/

  MoltenVK : bringing Vulkan to macOS/iOS 

* https://github.com/KhronosGroup/MoltenVK


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




