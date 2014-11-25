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




