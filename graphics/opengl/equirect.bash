# === func-gen- : graphics/opengl/equirect fgp graphics/opengl/equirect.bash fgn equirect fgh graphics/opengl src base/func.bash
equirect-source(){   echo ${BASH_SOURCE} ; }
equirect-edir(){ echo $(dirname $(equirect-source)) ; }
equirect-ecd(){  cd $(equirect-edir); }
equirect-dir(){  echo $LOCAL_BASE/env/graphics/opengl/equirect ; }
equirect-cd(){   cd $(equirect-dir); }
equirect-vi(){   vi $(equirect-source) ; }
equirect-env(){  elocal- ; }
equirect-usage(){ cat << EOU

Equirectangular Projection : 360 degree camera : spherical projection
=============================================================================

Trivial in a ray tracer, distinctly not with OpenGL
pipeline starting from vertices.

Background
------------

* https://stackoverflow.com/questions/44082298/how-to-make-360-video-output-in-opengl

* https://en.wikipedia.org/wiki/Equirectangular_projection

* https://en.wikipedia.org/wiki/Fisheye_lens#Fisheye_lens

* https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection

* https://en.wikipedia.org/wiki/List_of_map_projections

* https://en.wikipedia.org/wiki/Omnidirectional_camera

* https://en.wikipedia.org/wiki/360-degree_video

* https://en.wikipedia.org/wiki/Tissot's_indicatrix


360 Video CG Capture
--------------------------

* https://developers.google.com/vr/

* https://developers.google.com/vr/discover/360-degree-media


YouTube 360-degree videos : Upload instructions
----------------------------------------------------

* https://support.google.com/youtube/answer/6178631

Facebook 360 : Pyramid format
--------------------------------

* https://code.fb.com/virtual-reality/next-generation-video-encoding-techniques-for-360-video-and-vr/


:google:`equal area sphere mapping CMB`
------------------------------------------

HEALPix
~~~~~~~~

* https://healpix.sourceforge.io/pdf/intro.pdf

is a genuinely curvilinear partition of the sphere into exactly equal area quadri-
laterals of varying shape. The base-resolution comprises twelve pixels in three rings around
the poles and equator

* https://healpix.jpl.nasa.gov/html/intronode3.htm



* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5484980/

  A multi-resolution HEALPix data structure for spherically mapped point data

* ~/opticks_refs/multi_resolution_healpix.pdf

Hierarchical Equal Area iso-Latitude Pixelization (HEALPix), partitions the
sphere into twelve diamond-shaped equal-area base cells and then recursively
subdivides each cell into four diamond-shaped subcells, continuing to the
desired level of resolution. Twelve quadtrees, one associated with each base
cell, store the data records associated with that cell and its subcells.


SQT : Sphere Quadtree
~~~~~~~~~~~~~~~~~~~~~~~~

* https://ieeexplore.ieee.org/document/146380

  Rendering and managing spherical data with sphere quadtrees

The sphere quadtree (SQT), which is based on the recursive subdivision of
spherical triangles obtained by projecting the faces of an icosahedron onto a
sphere, is discussed. Most databases for spherically distributed data are not
structured in a manner consistent with their geometry. As a result, such
databases possess undesirable artifacts, including the introduction of tears in
the data when they are mapped onto a flat file system. Furthermore, it is
difficult to make queries about the topological relationship among the data
components without performing real arithmetic. The SQT eliminates some of these
problems. The SQT allows the representation of data at multiple levels and
arbitrary resolution. Efficient search strategies can be implemented for the
selection of data to be rendered or analyzed by a specific technique. Geometric
and topological consistency with the data are maintained.<>



Google Equi-Angular Cubemaps (EACs)
---------------------------------------

* https://en.wikipedia.org/wiki/360_video_projection

* https://blog.google/products/google-ar-vr/bringing-pixels-front-and-center-vr-video/

* https://youtube-eng.googleblog.com/2017/03/improving-vr-videos.html

* https://github.com/ytdl-org/youtube-dl/issues/15267

* https://github.com/ytdl-org/youtube-dl


Rendering Omni‐directional Stereo Content : Rays emanate from a circle of IPD diameter
------------------------------------------------------------------------------------------

* https://developers.google.com/vr/jump/rendering-ods-content.pdf

You have probably noticed that the ODS images at the beginning of this document
look similar to a conventional cylindrical or spherical (equirectangular)
panorama. Indeed, when the IPD goes to zero or if the scene is at infinity, ODS 
becomes equirectangular. Consequently, it can be used just like an
equirectangular image, that is, projected onto a sphere and re­rendered, and 
it’s compatible with existing panorama viewers.


Not so easy in OpenGL
------------------------

* https://community.khronos.org/t/lat-long-spherical-projection-using-vertex-shader/75517/7

* https://forum.openframeworks.cc/t/equirectangular-projection-shader/19937/6


ShaderToy
------------

* https://www.shadertoy.com/view/MlfSz7

  Equirectangular projection 
  // this is close to whats needed at first glance ... but no its only working 
  // thanks to the pre-baked texture 

* https://shadertoyunofficial.wordpress.com/2019/01/02/programming-tricks-in-shadertoy-glsl/


OpenGL Cube Mapping
---------------------


* http://antongerdelan.net/opengl/cubemaps.html


OpenGL Sphere Mapping
----------------------

* https://www.opengl.org/archives/resources/code/samples/advanced/advanced97/notes/node95.html#SECTION000113220000000000000



Dynamic Cube Mapping In OpenGL 
---------------------------------

* https://darrensweeney.net/2016/10/03/dynamic-cube-mapping-in-opengl/

* http://paulbourke.net/geometry/transformationprojection/


EOU
}
equirect-get(){
   local dir=$(dirname $(equirect-dir)) &&  mkdir -p $dir && cd $dir

}
