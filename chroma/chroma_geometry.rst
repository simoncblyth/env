Chroma Geometry Source Overview
=================================


where is geometry populated
-----------------------------

So where is `Geometry` populated::

    (chroma_env)delta:cuda blyth$ grep geometry_types.h *.*
    bvh.cu:#include "geometry_types.h"
    geometry.h:#include "geometry_types.h"    # device funcs that query the geometry, accessing nodes/triangles etc..


From python with `chroma/gpu/geometry.py:GPUGeometry` using `pycuda.gpuarray` and `chroma.gpu.tools.make_gpu_struct`



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


    (chroma_env)delta:chroma blyth$ find ../bin -type f -exec grep -H GPUGeometry {} \;

    ../bin/chroma-bvh:from chroma.gpu.geometry import GPUGeometry
    ../bin/chroma-bvh:    gpu_geometry = GPUGeometry(geometry)




GPUGeometry
~~~~~~~~~~~~~

GPU `Geometry` struct is constructed and populated from the below python using 

* `pycuda.gpuarray` 

  * http://documen.tician.de/pycuda/array.html

* `chroma.gpu.tools.make_gpu_struct`


`chroma/gpu/geometry.py`::

     13 class GPUGeometry(object):
     14     def __init__(self, geometry, wavelengths=None, print_usage=False, min_free_gpu_mem=300e6):
     15         if wavelengths is None:
     16             wavelengths = standard_wavelengths
     17 
     18         try:
     19             wavelength_step = np.unique(np.diff(wavelengths)).item()
     20         except ValueError:
     21             raise ValueError('wavelengths must be equally spaced apart.')
     22 
     23         geometry_source = get_cu_source('geometry_types.h')
     24         material_struct_size = characterize.sizeof('Material', geometry_source)
     25         surface_struct_size = characterize.sizeof('Surface', geometry_source)
     26         geometry_struct_size = characterize.sizeof('Geometry', geometry_source)
     27 
     ..

The materials/surfaces/mesh from the python geometry object are transferred over 
to the GPU and `Geometry` struct is created to hold the pointers.



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

* Is the below wavelength comment outdated ?

My impression was that the wavelengths used are held in the material/surface 
structs and interpolated as appropriate.::

     15 # all material/surface properties are interpolated at these
     16 # wavelengths when they are sent to the gpu
     17 standard_wavelengths = np.arange(60, 810, 20).astype(np.float32)
     18 
     19 class Mesh(object):
     20     "Triangle mesh object."
     21     def __init__(self, vertices, triangles, remove_duplicate_vertices=False):
     22         vertices = np.asarray(vertices, dtype=np.float32)
     23         triangles = np.asarray(triangles, dtype=np.int32)

Python side geometry

* `Geometry`, a detector_material and a list of Solids, rotations and displacements

  * `flatten` method determines global unique_materials, unique_surfaces from those for each solid

* `Solid`, attaches materials, surfaces, and colors to each triangle in the Mesh object argument
* `Mesh` , comprising arrays of vertices and triangles
* `Material`, with name and wavelength dependant property arrays:

  * refractive_index
  * absorption_length
  * scattering_length
  * reemission_prob
  * reemission_cdf
  * density
  * composition

* `Surface`, with name and model and wavelength dependant optical property arrays: 

  * detect/absort/reemit/reflect_diffuse/reflect_specular/eta/k/reemission_cdf/thickness/transmissive


Q: where all these properties getting set ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* not in the STL, thats a very simple list of vertices/triangles 

Chroma geometry construction currently done in "ad-hoc" python such as `chroma/demo/__init__.py`, 
Not out of some "standard" file format, like G4DAE COLLADA+metadata 


Chroma BVH class chroma/bvh/bvh.py
-----------------------------------

A bounding volume hierarchy for a triangle mesh.

For the purposes of Chroma, a BVH is a tree with the following properties:

* Each node consists of an axis-aligned bounding box, a child ID
  number, and a boolean flag indicating whether the node is a
  leaf.  The bounding box is represented as a lower and upper
  bound for each Cartesian axis.


chroma/cuda/geometry_types.h
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     46 struct Node
     47 {
     48     float3 lower;
     49     float3 upper;
     50     unsigned int child;
     51     unsigned int nchild;
     52 };

* All nodes are stored in a 1D array with the root node first.

* A node with a bounding box that has no surface area (upper and
  lower bounds equal for all axes) is a dummy node that should
  be ignored.  Dummy nodes are used to pad the tree to satisfy
  the fixed degree requirement described below, and have no
  children.

* If the node is a leaf, then the child ID number refers to the
  ID number of the triangle this node contains.

* If the node is not a leaf (an "inner" node), then the child ID
  number indicates the offset in the node array of the first
  child.  The other children of this node will be stored
  immediately after the first child.

* All inner nodes have the same number of children, called the
  "degree" (technically the "out-degree") of the tree.  This
  avoid the requirement to save the degree with the node.

* For simplicity, we also require nodes at the same depth
  in the tree to be contiguous, and the layers to be in order
  of increasing depth.

* All nodes satisfy the **bounding volume hierarchy constraint**:
  their bounding boxes contain the bounding boxes of all their
  children.

For space reasons, the BVH bounds are internally represented using
16-bit unsigned fixed point coordinates.  Normally, we would want
to hide that from you, but we would like to avoid rounding issues
and high memory usage caused by converting back and forth between
floating point and fixed point representations.  For similar
reasons, the node array is stored in a packed record format that
can be directly mapped to the GPU.  In general, you will not need
to manipulate the contents of the BVH node array directly.




chroma/cuda/mesh.h
--------------------

Stack based recursive tree walk::

     36 /* Finds the intersection between a ray and `geometry`. If the ray does
     37    intersect the mesh and the index of the intersected triangle is not equal
     38    to `last_hit_triangle`, set `min_distance` to the distance from `origin` to
     39    the intersection and return the index of the triangle which the ray
     40    intersected, else return -1. */
     41 __device__ int
     42 intersect_mesh(const float3 &origin, const float3& direction, Geometry *g,
     43            float &min_distance, int last_hit_triangle = -1)
     44 {
     45     int triangle_index = -1;
     46 





