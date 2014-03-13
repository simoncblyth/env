Chroma Issues
==============

.. contents:: :local:


EXECUTIVE SUMMARIES ONLY
------------------------

Use this page for:

* brief overviews of activities
* **KEEP IT BRIEF : LINK TO INVESTIGATIONS/NOTES ON OTHER PAGES**


Chroma CUDA Instability, GPU Panics
-------------------------------------

* :doc:`chroma_cuda_freeze`

Potential solutions:

* improve pycuda error handling :doc:`/pycuda/pycuda_error_handling`
* reduce work done by each kernel, maybe tweaking nblocks etc..
* operate headlessly in console mode to avoid GPU timeouts

Chroma Camera is uncontrollable
--------------------------------

* on rotating around, flip into other positions (maybe gimbal lock)


Chroma Surface model mismatch G4LogicalBoundarySurface
-------------------------------------------------------

G4LogicalBoundarySurface uses a pair of **ordered** PVs, 
so the surface is only relevent when stepping in from one direction
and not the other.

Chroma has only a single 8 bits for a surface index attached to 
each triangle. Solution may be to expand that to 2*8 bits.




 
