# === func-gen- : graphics/scene/scene fgp graphics/scene/scene.bash fgn scene fgh graphics/scene
scene-src(){      echo graphics/scene/scene.bash ; }
scene-source(){   echo ${BASH_SOURCE:-$(env-home)/$(scene-src)} ; }
scene-vi(){       vi $(scene-source) ; }
scene-env(){      elocal- ; }
scene-usage(){ cat << EOU

Scene
======

Overview
----------

There are literally 100s of 3D file formats for scene description. 
And probably there will be yet another .oks "OpticksScene" format too.. 
nevertherless can learn from the structure of existing ones. And also 
being able to convert from .oks into other formats will 
be made easier by knowing about them whilst devising oks.

Current favorite to look into is gltf as it is an emerging standard 
and is a progression from COLLADA.

Questions
~~~~~~~~~~~~~

* How to handle the CSG node tree structure ? 
* How to handle instances of repeated geometry and their transforms


Ingredients
------------

* nodes in a hierarchy 
* bounding boxes (global, local)
* local transforms
* global transforms
* geometry meshes


Eberly : 3D Game Engine Architecture (2005) : Regards Design of "Wild Magic 3"
-----------------------------------------------------------------------------------

Ch3 : Scene Graphs and Renderers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Spatial   
        parent link, transforms 

    Node(Spatial) 
        children 

    Geometry(Spatial)  
        only leaves, ie no children 


p156: 
   ... (Node) compositing of transforms is done thru depth-first 
   traversal of the parent-child tree. Each parent node provides
   its world transform to its child nodes in order for them to compute
   their world transforms

   [holding the composite transform at each level, avoids having to 
    composite all the way from the top for every leaf]

p157:
    changing local transforms at a node requires updating all the transforms for the
    subtree .. BUT also requires updates of bounds up the tree

p160 3.1.2 Spatial Hierarchy Design
    ... price for separation of concerns between Node(grouping) and Geometry 
    is that must introduce extra grouping nodes to hold geometry as geomerty cannot
    hold other geometry      
 
p163 3.1.3 Instancing
    Spatial hierarchy system is a tree structure

    * each tree node has a single parent, except for root with None.
    * acts a skeleton for the scene graph 

    Scene graph is an abstract graph (as opposed to a tree) because the object
    system supports sharing.  If an object is shared by two other objects, effectively 
    there are two instances of the first object (called instancing)

    ... instancing of nodes in the spatial hierarchy is **not allowed** : enforced
    by allowing Spatial to have only a single parent link 
    ... multiple parent DAGs are not possible 
   
    Rationale for "node tree" and not "node graph"... 
    consider the case of allowing node instancing

    ::

            House
           /     \ 
        Room1   Room2
           \     /
          Room-Contents

 
    * path from root to leaf has sequence of local transforms, the product
      of which is the world transform of the leaf node. But Room-contents 
      can be reached by 2 paths : so which transform is applicable ? Also
      where to persist the transform (in a tree can store directly in the node)

      Using DAG means you need to somehow dynamically create a tree in which to 
      keep the global transforms... also unclear how to specify a particular node..
      leads to complicated system of multiple parents 

    * much simpler to disallow internal node instancing, allow only instancing 
      of geometry (which cannot have children)

      #. nodes are very lightweight (just transforms and lists of child indices essentially) 
         its cheap to make copies of them 
      #. geometry is very heavy and really needs to not to be duplicated 

    ::

            House
           /     \ 
        Room1   Room2
          /        \
     Room1-        Room2-
     Contents      Contents
          \         /
       Shared-Geometry-Data



     


  

        
   
      

    



oks : OpticksScene format
----------------------------

Low level structure
~~~~~~~~~~~~~~~~~~~~~~

Directory format (rather than a file format) containing

* .txt lists 
* .json metadata
* .npy buffers

Needs to handle
~~~~~~~~~~~~~~~~

* instances without repeating info
* optional presence of mesh buffers, to act as polygonization cache
* optional presence of concatenated buffers, for caching 



gltf (recent) : aiming to be the JPEG of 3D
------------------------------------------------

* https://www.khronos.org/gltf
* https://github.com/KhronosGroup/glTF
* https://github.com/KhronosGroup/glTF#gltf-tools￼ 

  Lots of tools already for a very young "standard"


A new standard for 3D scenes is gaining momentum with support from graphics
industry leaders, potentially laying the groundwork for science fiction’s
“metaverse” to be realized.

The GL Transmission Format (glTF) from The Khronos Group, a computer graphics
industry standards body, could also put magnitudes more 3D content on the
Internet. The Khronos Group is responsible for a variety of technologies
critical to how computers show visuals. Standards include Vulkan, OpenGL, WebGL
and others. One of the latest is glTF, designed to streamline the way 3D
content is transmitted and loaded across any device. JPEG helped lead to an
explosion in the way people make and use images and glTF could do that for 3D
scenes.

* https://www.khronos.org/assets/uploads/developers/library/2017-glTF-webinar/glTF-Webinar_Feb17.pdf







Ray Tracer Scene Description, file formats
--------------------------------------------

* Which Ray tracers have separate scene description languages ? 
* Any kind of standards ?

* https://en.m.wikipedia.org/wiki/List_of_ray_tracing_software



Describing Robots
---------------------

* http://sdformat.org



Scene Description Language Examples
------------------------------------

* https://en.m.wikipedia.org/wiki/Scene_description_language

* Pov-ray while loops

2008 NCSA review of 3D file formats (~140 different formats!)
----------------------------------------------------------------

* https://www.archives.gov/files/applied-research/ncsa/8-an-overview-of-3d-data-content-file-formats-and-viewers.pdf


OpenSCAD CSG file format
-------------------------

See openscad-


Alembic
-----------

* http://www.alembic.io

Alembic is an open computer graphics interchange framework. Alembic distills
complex, animated scenes into a non-procedural, application-independent set of
baked geometric results.

Relation to RenderMan ?

* https://github.com/alembic/alembic/tree/master/prman/Procedural


X3D (very old standard)
--------------------------

X3D is a royalty-free ISO standard XML-based file format for representing 3D
computer graphics. It is successor to the Virtual Reality Modeling Language
(VRML).


RenderMan 
-----------

Moved to renderman-


Autodesk SDL
--------------

* http://www.autodesk.com/techpubs/studiotools/13/PDFs/SDL


.mi : NVIDIA mental-ray, iray 
--------------------------------

* https://blog.mentalray.com/2014/10/01/let-mi-export/

For those who use mental ray Standalone to render on a remote computer or in a
render farm, the creation of a scene file in the proprietary .mi format is a
necessary step. Most content creation tools are able to “echo” the full scene
into a .mi representation on disk using mental ray’s built-in capability. But
mental ray for Maya implements a much more flexible approach with extra
functionality to ease render pipeline integration. In this post we would like
to take a closer look at some of these features.

iray
------

* http://blog.irayrender.com

* http://blog.irayrender.com/post/157353002021/maxwell-gpuswindows-10-performance-improvement

  "Optimize for Compute Performance" 
  February 17, 2017






EOU
}
scene-dir(){ echo $(local-base)/env/graphics/scene/graphics/scene-scene ; }
scene-cd(){  cd $(scene-dir); }
scene-mate(){ mate $(scene-dir) ; }
scene-get(){
   local dir=$(dirname $(scene-dir)) &&  mkdir -p $dir && cd $dir

}
