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


Some fiddly splitting of data between GPU and CPU ?::

    196 def Mapped(array):
    197     '''Analog to pycuda.driver.InOut(), but indicates this array
    198     is memory mapped to the device space and should not be copied.
    199 
    200     To simplify coding, Mapped() will pass anything with a gpudata
    201     member, like a gpuarray, through unchanged.
    202     '''
    203     if hasattr(array, 'gpudata'):
    204         return array
    205     else:
    206         return np.intp(array.base.get_device_pointer())

    In [91]: np.intp
    Out[91]: numpy.int32


Heavy lifting of getting geometry onto GPU::

    134         self.vertices = mapped_empty(shape=len(geometry.mesh.vertices),
    135                                      dtype=ga.vec.float3,
    136                                      write_combined=True)
    137         self.triangles = mapped_empty(shape=len(geometry.mesh.triangles),
    138                                       dtype=ga.vec.uint3,
    139                                       write_combined=True)
    140         self.vertices[:] = to_float3(geometry.mesh.vertices)
    141         self.triangles[:] = to_uint3(geometry.mesh.triangles)
    142 

    143         self.world_origin = ga.vec.make_float3(*geometry.bvh.world_coords.world_origin)
    144         self.world_scale = np.float32(geometry.bvh.world_coords.world_scale)

    145 
    146         material_codes = (((geometry.material1_index & 0xff) << 24) |
    147                           ((geometry.material2_index & 0xff) << 16) |
    148                           ((geometry.surface_index & 0xff) << 8)).astype(np.uint32)
    149         self.material_codes = ga.to_gpu(material_codes)

    ///   packing 3 single byte indices (0:255) into unint32 with low byte empty

    In [110]: "0x%x" %  (((0xaa & 0xff) << 24) | ((0xbb & 0xff) << 16) |  ((0xcc & 0xff) << 8))
    Out[110]: '0xaabbcc00'



    150         colors = geometry.colors.astype(np.uint32)
    151         self.colors = ga.to_gpu(colors)

    152         self.solid_id_map = ga.to_gpu(geometry.solid_id.astype(np.uint32))
    153 


    154         # Limit memory usage by splitting BVH into on and off-GPU parts
    155         gpu_free, gpu_total = cuda.mem_get_info()
    156         node_array_usage = geometry.bvh.nodes.nbytes
    157 
    158         # Figure out how many elements we can fit on the GPU,
    159         # but no fewer than 100 elements, and no more than the number of actual nodes
    160         n_nodes = len(geometry.bvh.nodes)
    161         split_index = min(
    162             max(int((gpu_free - min_free_gpu_mem) / geometry.bvh.nodes.itemsize),100),
    163             n_nodes
    164             )
    165 
    166         self.nodes = ga.to_gpu(geometry.bvh.nodes[:split_index])
    167         n_extra = max(1, (n_nodes - split_index)) # forbid zero size
    168         self.extra_nodes = mapped_empty(shape=n_extra,
    169                                         dtype=geometry.bvh.nodes.dtype,
    170                                         write_combined=True)
    171         if split_index < n_nodes:
    172             logger.info('Splitting BVH between GPU and CPU memory at node %d' % split_index)
    173             self.extra_nodes[:] = geometry.bvh.nodes[split_index:]
    174 
    175         # See if there is enough memory to put the and/ortriangles back on the GPU
    176         gpu_free, gpu_total = cuda.mem_get_info()
    177         if self.triangles.nbytes < (gpu_free - min_free_gpu_mem):
    178             self.triangles = ga.to_gpu(self.triangles)
    179             logger.info('Optimization: Sufficient memory to move triangles onto GPU')
    180 
    181         gpu_free, gpu_total = cuda.mem_get_info()
    182         if self.vertices.nbytes < (gpu_free - min_free_gpu_mem):
    183             self.vertices = ga.to_gpu(self.vertices)
    184             logger.info('Optimization: Sufficient memory to move vertices onto GPU')
    185 
    186         self.gpudata = make_gpu_struct(geometry_struct_size,
    187                                        [Mapped(self.vertices),
    188                                         Mapped(self.triangles),
    189                                         self.material_codes,
    190                                         self.colors, self.nodes,
    191                                         Mapped(self.extra_nodes),
    192                                         self.material_pointer_array,
    193                                         self.surface_pointer_array,
    194                                         self.world_origin,
    195                                         self.world_scale,
    196                                         np.int32(len(self.nodes))])


material and surface indices packed into material_codes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hmm, there are loadsa materials and surfaces, so whats this `material_codes` qty ?::

    145 
    146         material_codes = (((geometry.material1_index & 0xff) << 24) |
    147                           ((geometry.material2_index & 0xff) << 16) |
    148                           ((geometry.surface_index & 0xff) << 8)).astype(np.uint32)
    149         self.material_codes = ga.to_gpu(material_codes)

    ///   packing 3 single byte indices (0:255) into unint32 with low byte empty

    In [110]: "0x%x" %  (((0xaa & 0xff) << 24) | ((0xbb & 0xff) << 16) |  ((0xcc & 0xff) << 8))
    Out[110]: '0xaabbcc00'

    In [130]: (np.array([(1<<8)-1, (1<<8), (1<<8)+1 ]) & 0xff ).astype(np.uint32) 
    Out[130]: array([255,   0,   1], dtype=uint32)



chroma/geometry.py::

    301     def flatten(self):
    302         """
    303         Create the flat list of triangles (and triangle properties)
    304         from the list of solids in this geometry.
    305 
    306         This does not build the BVH!  If you want to use the geometry
    307         for rendering or simulation, you should call build() instead.
    308         """
    309 
    310         # Don't run this function twice!
    311         if hasattr(self, 'mesh'):
    312             return
    313 
    314         nv = np.cumsum([0] + [len(solid.mesh.vertices) for solid in self.solids])
    315         nt = np.cumsum([0] + [len(solid.mesh.triangles) for solid in self.solids])
    316 
    317         vertices = np.empty((nv[-1],3), dtype=np.float32)
    318         triangles = np.empty((nt[-1],3), dtype=np.uint32)
    319 
    320 
    321         logger.info('Flattening detector mesh...')
    322         logger.info('  triangles: %d' % len(triangles))
    323         logger.info('  vertices:  %d' % len(vertices))
    324 
    325 
    326         for i, solid in enumerate(self.solids):
    327             vertices[nv[i]:nv[i+1]] = \
    328                 np.inner(solid.mesh.vertices, self.solid_rotations[i]) + self.solid_displacements[i]
    329             triangles[nt[i]:nt[i+1]] = solid.mesh.triangles + nv[i]
    330 
    331         # Different solids are very unlikely to share vertices, so this goes much faster
    332         self.mesh = Mesh(vertices, triangles, remove_duplicate_vertices=False)
    333 
    334         self.colors = np.concatenate([solid.color for solid in self.solids])
    335 
    336         self.solid_id = np.concatenate([filled_array(i, shape=len(solid.mesh.triangles), dtype=np.uint32) for i, solid in enumerate(self.solids)])
    337 
    338         self.unique_materials = list(np.unique(np.concatenate([solid.unique_materials for solid in self.solids])))
    339 
    340         material_lookup = dict(zip(self.unique_materials, range(len(self.unique_materials))))
    //
    //          map from material to its unique index
    //
    341 
    342         self.material1_index = np.concatenate([solid.material1_indices(material_lookup) for solid in self.solids])
    343 
    //          array of material1 indices of every triangle in every solid 
    //
    344         self.material2_index = np.concatenate([solid.material2_indices(material_lookup) for solid in self.solids])
    345 
    346         self.unique_surfaces = list(np.unique(np.concatenate([solid.unique_surfaces for solid in self.solids])))
    347 
    348         surface_lookup = dict(zip(self.unique_surfaces, range(len(self.unique_surfaces))))
    349 
    350         self.surface_index = np.concatenate([solid.surface_indices(surface_lookup) for solid in self.solids])
    351 
    352         try:
    353             self.surface_index[self.surface_index == surface_lookup[None]] = -1
    354         except KeyError:
    355             pass
    356 

chroma/cuda/photon.h::

     79 __device__ void
     80 fill_state(State &s, Photon &p, Geometry *g)
     81 {
     82     p.last_hit_triangle = intersect_mesh(p.position, p.direction, g,
     83                                          s.distance_to_boundary,
     84                                          p.last_hit_triangle);
     85 
     86     if (p.last_hit_triangle == -1) {
     87         p.history |= NO_HIT;
     88         return;
     89     }
     90 
     91     Triangle t = get_triangle(g, p.last_hit_triangle);
     92 
     93     unsigned int material_code = g->material_codes[p.last_hit_triangle];
     94 
     95     int inner_material_index = convert(0xFF & (material_code >> 24));
     96     int outer_material_index = convert(0xFF & (material_code >> 16));
     97     s.surface_index = convert(0xFF & (material_code >> 8));


     In [135]: "0x%x" % (0xff & (0xaabbcc00 >> 24 ))
     Out[135]: '0xaa'

     In [136]: "0x%x" % (0xff & (0xaabbcc00 >> 16 ))
     Out[136]: '0xbb'

     In [137]: "0x%x" % (0xff & (0xaabbcc00 >> 8 ))
     Out[137]: '0xcc'

     98 
     99     float3 v01 = t.v1 - t.v0;
     00     float3 v12 = t.v2 - t.v1;
     01 
     02     s.surface_normal = normalize(cross(v01, v12));
     03 
     04     Material *material1, *material2;
     05     if (dot(s.surface_normal,-p.direction) > 0.0f) {
     06         // outside to inside
     07         material1 = g->materials[outer_material_index];
     08         material2 = g->materials[inner_material_index];
     09 
     10         s.inside_to_outside = false;
     11     }
     12     else {
     13         // inside to outside
     14         material1 = g->materials[inner_material_index];
     15         material2 = g->materials[outer_material_index];
     16         s.surface_normal = -s.surface_normal;
     17 
     18         s.inside_to_outside = true;
     19     }
     20 
     21     s.refractive_index1 = interp_property(material1, p.wavelength,
     22                                           material1->refractive_index);
     23     s.refractive_index2 = interp_property(material2, p.wavelength,
     24                                           material2->refractive_index);
     25     s.absorption_length = interp_property(material1, p.wavelength,
     26                                           material1->absorption_length);
     27     s.scattering_length = interp_property(material1, p.wavelength,
     28                                           material1->scattering_length);
     29     s.reemission_prob = interp_property(material1, p.wavelength,
     30                                         material1->reemission_prob);
     31 
     32     s.material1 = material1;
     33 } // fill_state




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





