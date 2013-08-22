Bounding Volume Hierarchy (BVH)
=====================================

* :google:`bounding volume hierarchy`


Background
-----------

* http://en.wikipedia.org/wiki/Bounding_volume_hierarchy

   * With such a hierarchy in place, during collision testing, children do not
     have to be examined if their parent volumes are not intersected.

* http://www.3dmuve.com/3dmblog/?p=182

   * A brief tutorial on what BVH are and how to implement them


* https://developer.nvidia.com/content/thinking-parallel-part-i-collision-detection-gpu
* https://developer.nvidia.com/content/thinking-parallel-part-ii-tree-traversal-gpu

     *  The idea is to traverse the hierarchy in a top-down manner, starting from the
        root. For each node, we first check whether its bounding box overlaps with the
        query. If not, we know that none of the underlying leaf nodes will overlap it
        either, so we can skip the entire subtree. Otherwise, we check whether the node
        is a leaf or an internal node. If it is a leaf, we report a potential collision
        with the corresponding object. If it is an internal node, we proceed to test
        each of its children in a recursive fashion.

    *   Instead of launching one thread per object, as we did previously, we are now
        launching one thread per leaf node. This does not affect the behavior of the
        kernel, since each object will still get processed exactly once. However, it
        changes the ordering of the threads to minimize both execution and data
        divergence. The total execution time is now 0.43 milliseconds?this trivial
        change improved the performance of our algorithm by another 2x!

        HUH ? 


* https://developer.nvidia.com/content/thinking-parallel-part-iii-tree-construction-gpu

* http://en.wikipedia.org/wiki/Space-filling_curve

* http://en.wikipedia.org/wiki/Z-order_curve (aka Morton order)

A function which maps multidimensional data to one dimension while preserving locality of the data points 

The z-value of a point in multidimensions is simply calculated by interleaving
the binary representations of its coordinate values. Once the data are sorted
into this ordering, any one-dimensional data structure can be used such as
binary search trees, B-trees, skip lists or (with low significant bits
truncated) hash tables. The resulting ordering can equivalently be described as
the order one would get from a depth-first traversal of a quadtree; because of
its close connection with quadtrees, the Z-ordering can be used to efficiently
construct quadtrees and related higher dimensional data structures.



Chroma BVH class
-------------------

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


