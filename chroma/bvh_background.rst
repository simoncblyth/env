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



