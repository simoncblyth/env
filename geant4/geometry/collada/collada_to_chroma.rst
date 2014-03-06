Collada To Chroma
==================

chroma-cam
-----------

First run
~~~~~~~~~~

::

    (chroma_env)delta:~ blyth$ chroma-cam $ENV_HOME/geant4/geometry/materials/g4_00.dae
    WARNING:env.geant4.geometry.collada.daenode:failed to find parent for   top.0             -  (failure expected only for root node)
    INFO:chroma:Flattening detector mesh...
    INFO:chroma:  triangles: 2483650
    INFO:chroma:  vertices:  1264049
    INFO:chroma:Building new BVH using recursive grid algorithm.
    Expanding 22240 parent nodes
    Merging 2483650 nodes to 79179 parents
    Expanding 3675 parent nodes
    Merging 220475 nodes to 61739 parents
    Expanding 235 parent nodes
    Merging 65805 nodes to 18559 parents
    Expanding 8 parent nodes
    Merging 18806 nodes to 4597 parents
    Merging 4605 nodes to 1196 parents
    Merging 1196 nodes to 300 parents
    Merging 300 nodes to 95 parents
    Expanding 1 parent nodes
    Merging 95 nodes to 30 parents
    Merging 31 nodes to 8 parents
    Merging 8 nodes to 2 parents
    Merging 2 nodes to 1 parent
    INFO:chroma:BVH generated in 7.2 seconds.
    INFO:chroma:Saving BVH (c600c30494ebdd3c9fed0474e09f9a80:default) to cache.
    INFO:chroma:loaded geometry <chroma.geometry.Geometry object at 0x116824cd0> 
    INFO:chroma:starting view [1024, 576] 
    INFO:chroma:create Camera 
    INFO:chroma:Camera.__init__
    INFO:chroma:Camera.__init__ done
    INFO:chroma:_run Camera 
    /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/characterize.py:40: UserWarning: The CUDA compiler succeeded, but said the following:
    kernel.cu(45): warning: variable "NCHILD_MASK" was declared but never referenced

    kernel.cu(45): warning: variable "NCHILD_MASK" was declared but never referenced


      """ % (preamble, type_name), no_extern_c=True)
    Traceback (most recent call last):
      File "/usr/local/env/chroma_env/bin/chroma-cam", line 8, in <module>
        execfile(__file__)
      File "/usr/local/env/chroma_env/src/chroma/bin/chroma-cam", line 37, in <module>
        view_nofork(geometry, size)
      File "/usr/local/env/chroma_env/src/chroma/chroma/camera.py", line 849, in view_nofork
        camera._run()
      File "/usr/local/env/chroma_env/src/chroma/chroma/camera.py", line 639, in _run
        self.init_gpu()
      File "/usr/local/env/chroma_env/src/chroma/chroma/camera.py", line 87, in init_gpu
        self.gpu_geometry = gpu.GPUGeometry(self.geometry)
      File "/usr/local/env/chroma_env/src/chroma/chroma/gpu/geometry.py", line 41, in __init__
        raise Exception('one or more triangles is missing a material.')
    Exception: one or more triangles is missing a material.
    -------------------------------------------------------------------
    PyCUDA ERROR: The context stack was not empty upon module cleanup.
    -------------------------------------------------------------------
    A context was still active when the context stack was being
    cleaned up. At this point in our execution, CUDA may already
    have been deinitialized, so there is no way we can finish
    cleanly. The program will be aborted now.
    Use Context.pop() to avoid this problem.
    -------------------------------------------------------------------
    Abort trap: 6
    (chroma_env)delta:~ blyth$ 



After avoid None material
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    (chroma_env)delta:~ blyth$ chroma-cam $ENV_HOME/geant4/geometry/materials/g4_00.dae
    WARNING:env.geant4.geometry.collada.daenode:failed to find parent for   top.0             -  (failure expected only for root node)
    WARNING:env.geant4.geometry.collada.collada_to_chroma:setting parent_material to __dd__Materials__Vacuum0xaf1d298 as parent is None for node top.0 
    INFO:chroma:Flattening detector mesh...
    INFO:chroma:  triangles: 2483650
    INFO:chroma:  vertices:  1264049
    INFO:chroma:Loading BVH "default" for geometry from cache.
    INFO:chroma:loaded geometry <chroma.geometry.Geometry object at 0x10db83150> 
    INFO:chroma:starting view [1024, 576] 
    INFO:chroma:create Camera 
    INFO:chroma:Camera.__init__
    INFO:chroma:Camera.__init__ done
    INFO:chroma:_run Camera 
    Traceback (most recent call last):
      File "/usr/local/env/chroma_env/bin/chroma-cam", line 8, in <module>
        execfile(__file__)
      File "/usr/local/env/chroma_env/src/chroma/bin/chroma-cam", line 37, in <module>
        view_nofork(geometry, size)
      File "/usr/local/env/chroma_env/src/chroma/chroma/camera.py", line 849, in view_nofork
        camera._run()
      File "/usr/local/env/chroma_env/src/chroma/chroma/camera.py", line 639, in _run
        self.init_gpu()
      File "/usr/local/env/chroma_env/src/chroma/chroma/camera.py", line 87, in init_gpu
        self.gpu_geometry = gpu.GPUGeometry(self.geometry)
      File "/usr/local/env/chroma_env/src/chroma/chroma/gpu/geometry.py", line 43, in __init__
        refractive_index = interp_material_property(wavelengths, material.refractive_index)
      File "/usr/local/env/chroma_env/src/chroma/chroma/gpu/geometry.py", line 35, in interp_material_property
        return np.interp(wavelengths, property[:,0], property[:,1]).astype(np.float32)
    TypeError: 'NoneType' object has no attribute '__getitem__'
    -------------------------------------------------------------------
    PyCUDA ERROR: The context stack was not empty upon module cleanup.
    -------------------------------------------------------------------
    A context was still active when the context stack was being
    cleaned up. At this point in our execution, CUDA may already
    have been deinitialized, so there is no way we can finish
    cleanly. The program will be aborted now.
    Use Context.pop() to avoid this problem.
    -------------------------------------------------------------------
    Abort trap: 6



