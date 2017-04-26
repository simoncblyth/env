# === func-gen- : graphics/renderman/renderman fgp graphics/renderman/renderman.bash fgn renderman fgh graphics/renderman
renderman-src(){      echo graphics/renderman/renderman.bash ; }
renderman-source(){   echo ${BASH_SOURCE:-$(env-home)/$(renderman-src)} ; }
renderman-vi(){       vi $(renderman-source) ; }
renderman-env(){      elocal- ; }
renderman-usage(){ cat << EOU

Renderman
============




Terminology 

* https://renderman.pixar.com/view/renderman-prman-the-rispec-and-renderman-studio

* https://en.m.wikipedia.org/wiki/RenderMan_Interface_Specification
* https://renderman.pixar.com/products/rispec/rispec_pdf/RISpec3_2.pdf
* ~/opticks_refs/RISpec3_2.pdf (226 pages)

* http://www.graphics.stanford.edu/courses/cs448-05-winter/papers/course09.pdf
* ~/opticks_refs/RenderMan_Theory_And_Practice_2003_siggraph_course09.pdf


The RenderMan Interface Specification, or RISpec in short, is an open API
developed by Pixar Animation Studios to describe three-dimensional scenes and
turn them into digital photorealistic images. It includes the RenderMan Shading
Language.

* RISpec uses ANSI C language binding


RIB (RenderMan Interface Bytestream) is the serialization 
of the model built with the ANSI C 

* https://renderman.pixar.com/view/rib-scene-description


Intro : In Depth Renderman
------------------------------

* http://joomla.renderwiki.com/joomla/index.php?option=com_content&view=article&id=239&Itemid=213

The aim of this article is to discuss why prman or other renderman compliant
renderers are so famous and what are the cons and pros to use them. So first we
will see how they work and where they differ from other renderers.


Renderman Raytrace
~~~~~~~~~~~~~~~~~~~~~

* http://joomla.renderwiki.com/joomla/index.php?option=com_content&view=article&id=271

Raytracing is always a problem in renderman. This is not because it cannot
raytrace. And it works fine in small scenes. No problem. But as soon as you
come to scenes that are close to production scenes with thousends of objects
and thousends of textures, a lot of displacement and other junk, you will get
problems.

PRman and rman studio offer some controls to reduce the raytrace memory
demands. With the procedural cache option, you can set the upper limit for
procedurals. As soon as this limit is reached, prman tries to remove the least
used archives including all connected data from the memory and will reload it
on demand. Of course this increases renderime because a lot of geometry has to
be reloaded and retesselated, but ir works.



Free Non-Commercial RenderMan FAQ
------------------------------------

* https://renderman.pixar.com/view/dp25849



RenderMan procprims for handling lots of data 
-----------------------------------------------

* https://scicomp.stackexchange.com/questions/14824/rendering-huge-data-with-renderman-interface-bytestream

::

    The approach is not to generate the RIB on disk or generate all the RIB in one step.

    For such large amount of data, the first step is to organize it in some
    hierarchy fashion so that you have an Octree like bounding box structures. This
    is important from a memory foot print perspective.

    With the above, you would use the Procedural Geometry procedural approach
    to emit small amount of geometry when the rendering bucket hits a bounding box
    containing your geometry (hence the importance of hierarchy bounding boxes).
    Once a rendering bucket is done, the geometries are discard keeping the memory
    footprint in check.

    Note that the above is just the documentation from Pixar's PRMan (commercial)
    but is implemented in open source RenderMan renderer like Aqsis.


* https://renderman.pixar.com/resources/RenderMan_20/proceduralPrimitives.html  "procprims"


::

    RenderMan procedural primitives (procprims, for short) are user-provided
    subroutines that can be called upon to generate geometry (or issue other Ri
    requests) during the process of rendering. The advantage of procprims over
    concrete (RIB-based) primitives is that they can represent complex geometry
    very efficiently. Procprims can produce incredible geometric complexity from a
    small number of inputs, and we can defer this data amplification until the
    renderer is prepared to handle it. This may make it possible to render much
    more complex scenes than would be possible with a non-procedural
    representation. Procprims can also be thought of as units of memory management.
    The renderer can elect to unload all concrete geometry associated with a
    procprim knowing that the geometry can be regenerated should there be further
    need for it.

    To describe procedural primitives to RenderMan, use either RiProcedural or
    RiProcedural2.






EOU
}
renderman-dir(){ echo $(local-base)/env/graphics/renderman/graphics/renderman-renderman ; }
renderman-cd(){  cd $(renderman-dir); }
renderman-mate(){ mate $(renderman-dir) ; }
renderman-get(){
   local dir=$(dirname $(renderman-dir)) &&  mkdir -p $dir && cd $dir

}
