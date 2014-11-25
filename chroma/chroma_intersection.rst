Chroma Intersection
=====================

Box intersection with axis aligned photons
-----------------------------------------------

A problem was found with chroma handling of axis 
aligned photons (coming from MockNuWa). 
Checking with `g4daeview.py --nopropagate` see the 
problem is more specifically with vertical photons.

They result in kernels never completing, when running with GUI 
these cause terminations. When running in `>console`
mode my patience never lasted long enough.

From CUDA GDB-ing they enter `fill_state` and never
emerge getting stuck in `intersect_mesh/intersect_box`.  
The CUDA GDB session that revealed this is documented at 

* :doc:`cuda/cuda_gdb`

As the intersection code deals in reciprocal 
directions, clearly axis alignment results 
in infinities.


* http://tavianator.com/2011/05/fast-branchless-raybounding-box-intersections/


CUDA float operations IEEE 754 
--------------------------------


* https://devtalk.nvidia.com/default/topic/498671/how-to-generate-inf-and-inf-without-warning-does-anybody-know-that-/


intersect_mesh
----------------

::

    (chroma_env)delta:~ blyth$ RANGE=0:32 MockNuWa 1


Checking typical behaviour of `chroma/chroma/cuda/mesh.h`::

    2014-11-25 16:37:12,973 INFO    chroma.gpu.photon_hit:204 nwork 32 step 0 max_steps 30 nsteps 30 
    [Launch of CUDA Kernel 18 (propagate_hit<<<(1,1,1),(64,1,1)>>>) on Device 0]
    node gets: 906 triangle count: 190 maxcurr: 23 hitsame: 0 
    node gets: 788 triangle count: 100 maxcurr: 15 hitsame: 0 
    node gets: 365 triangle count: 49 maxcurr: 15 hitsame: 0 
    node gets: 764 triangle count: 160 maxcurr: 18 hitsame: 0 
    node gets: 1224 triangle count: 248 maxcurr: 20 hitsame: 0 
    node gets: 757 triangle count: 115 maxcurr: 16 hitsame: 0 
    node gets: 372 triangle count: 42 maxcurr: 14 hitsame: 0 
    node gets: 991 triangle count: 203 maxcurr: 19 hitsame: 0 
    node gets: 316 triangle count: 26 maxcurr: 14 hitsame: 0 
    node gets: 342 triangle count: 28 maxcurr: 15 hitsame: 0 
    node gets: 765 triangle count: 82 maxcurr: 14 hitsame: 0 
    node gets: 209 triangle count: 4 maxcurr: 12 hitsame: 0 
    node gets: 666 triangle count: 120 maxcurr: 14 hitsame: 0 
    node gets: 183 triangle count: 6 maxcurr: 11 hitsame: 0 
    node gets: 198 triangle count: 8 maxcurr: 9 hitsame: 0 
    node gets: 205 triangle count: 8 maxcurr: 8 hitsame: 0 
    node gets: 964 triangle count: 266 maxcurr: 25 hitsame: 0 
    node gets: 188 triangle count: 4 maxcurr: 9 hitsame: 0 
    node gets: 188 triangle count: 4 maxcurr: 9 hitsame: 0 
    node gets: 186 triangle count: 8 maxcurr: 9 hitsame: 0 
    node gets: 217 triangle count: 8 maxcurr: 8 hitsame: 0 
    node gets: 184 triangle count: 4 maxcurr: 9 hitsame: 0 
    node gets: 390 triangle count: 54 maxcurr: 10 hitsame: 0 
    node gets: 1082 triangle count: 240 maxcurr: 18 hitsame: 0 
    node gets: 662 triangle count: 70 maxcurr: 16 hitsame: 0 
    node gets: 241 triangle count: 4 maxcurr: 9 hitsame: 0 
    node gets: 390 triangle count: 36 maxcurr: 13 hitsame: 0 
    node gets: 503 triangle count: 49 maxcurr: 13 hitsame: 0 
    node gets: 213 triangle count: 8 maxcurr: 9 hitsame: 0 
    node gets: 157 triangle count: 4 maxcurr: 8 hitsame: 0 
    node gets: 191 triangle count: 6 maxcurr: 11 hitsame: 0 
    node gets: 335 triangle count: 20 maxcurr: 12 hitsame: 0 
    node gets: 507 triangle count: 89 maxcurr: 11 hitsame: 1 
    node gets: 291 triangle count: 27 maxcurr: 8 hitsame: 1 
    node gets: 243 triangle count: 11 maxcurr: 7 hitsame: 1 
    node gets: 159 triangle count: 3 maxcurr: 9 hitsame: 1 
    node gets: 460 triangle count: 86 maxcurr: 11 hitsame: 1 


After a few seconds of kernel run on single MOCK vertical photon::

    (chroma_env)delta:~ blyth$ RANGE=0:1 MockNuWa MOCK 

    cuda-gdb) p count
    $1 = 110695
    (cuda-gdb) p tri_count
    $2 = 86518
    (cuda-gdb) p maxcurr
    $3 = <value optimized out>
    (cuda-gdb) p hitsame
    $4 = 0
    (cuda-gdb) 


Its as if the x and y position of the photon is not coming into play, so 
are intersecting with most everythig in the tree ?::

    (cuda-gdb) p node
    $5 = {lower = {x = -22439.8477, y = -797443.5, z = -3032.02246}, upper = {x = -22425.7305, y = -797436.625, z = -3004.49121}, child = 1852655, nchild = 0}
    (cuda-gdb) c
    Continuing.

    Breakpoint 1, intersect_mesh(const float3 * @generic, const float3 * @generic, Geometry * @generic, float * @generic, int) (origin=0x3fffbb0, direction=0x3fffbbc, g=0x1000000, min_distance=0x3fffc1c, last_hit_triangle=-1) at mesh.h:83
    83                  if (node.nchild == 0) { /* leaf node */
    (cuda-gdb) p node
    $6 = {lower = {x = -22426.6699, y = -797442.5, z = -3032.02246}, upper = {x = -22425.4941, y = -797438.5, z = -3004.25586}, child = 1852751, nchild = 0}
    (cuda-gdb) 





