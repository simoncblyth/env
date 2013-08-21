Chroma Geometry Source Overview
=================================

chroma/cuda/geometry_types.h
------------------------------

::

     46 struct Node
     47 {
     48     float3 lower;
     49     float3 upper;
     50     unsigned int child;
     51     unsigned int nchild;
     52 };
     53 
     54 struct Geometry
     55 {
     56     float3 *vertices;
     57     uint3 *triangles;
     58     unsigned int *material_codes;
     59     unsigned int *colors;
     60     uint4 *primary_nodes;
     61     uint4 *extra_nodes;
     62     Material **materials;
     63     Surface **surfaces;
     64     float3 world_origin;
     65     float world_scale;
     66     int nprimary_nodes;
     67 };


* http://stackoverflow.com/a/4838734

chroma/gpu/geometry.py
------------------------

GPUGeometry class that converts into GPU side geometry using CUDA types from `geometry_types.h`

::

    simon:chroma blyth$ find . -name '*.py' -exec grep -H GPUGeometry {} \;
    ./camera.py:        self.gpu_geometry = gpu.GPUGeometry(self.geometry)
    ./camera.py:                gpu_geometry = gpu.GPUGeometry(geometry, print_usage=False)
    ./camera.py:        gpu_geometry = gpu.GPUGeometry(geometry)
    ./gpu/detector.py:from chroma.gpu.geometry import GPUGeometry
    ./gpu/detector.py:class GPUDetector(GPUGeometry):
    ./gpu/detector.py:        GPUGeometry.__init__(self, detector, wavelengths=wavelengths, print_usage=False)
    ./gpu/geometry.py:class GPUGeometry(object):
    ./sim.py:            self.gpu_geometry = gpu.GPUGeometry(detector)


chroma/loader.py
------------------

`def load_geometry_from_string`

::

     28       "filename.stl" or "filename.stl.bz2" - Create a geometry from a
     29           3D mesh on disk.  This model will not be cached, but the
     30           BVH can be, depending on whether update_bvh_cache is True.


chroma/stl.py
---------------

Parse STL files (simple format of vertices and triangles) into Mesh objects.


chroma/geometry.py
--------------------

`Mesh` object, comprising arrays of vertices and triangles


chroma/bvh/bvh.py
--------------------





