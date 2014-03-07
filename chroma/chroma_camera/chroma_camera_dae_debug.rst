Chroma Camera DAE Debug 
==========================

Initial Shakedown
------------------

First Run
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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



Setting defaults
~~~~~~~~~~~~~~~~~~

Succeed to visualize via Chroma, but due to hugeness of world volume navigation is near impossible.

::

    (chroma_env)delta:~ blyth$ chroma-cam $ENV_HOME/geant4/geometry/materials/g4_00.dae
    WARNING:env.geant4.geometry.collada.daenode:failed to find parent for   top.0             -  (failure expected only for root node)
    WARNING:env.geant4.geometry.collada.collada_to_chroma:setting parent_material to __dd__Materials__Vacuum0xaf1d298 as parent is None for node top.0 
    INFO:chroma:Flattening detector mesh...
    INFO:chroma:  triangles: 2483650
    INFO:chroma:  vertices:  1264049
    INFO:chroma:Loading BVH "default" for geometry from cache.
    INFO:chroma:loaded geometry <chroma.geometry.Geometry object at 0x11954f190> 
    INFO:chroma:starting view [1024, 576] 
    INFO:chroma:create Camera 
    INFO:chroma:Camera.__init__
    INFO:chroma:Camera.__init__ done
    INFO:chroma:_run Camera 
    INFO:chroma:Optimization: Sufficient memory to move triangles onto GPU
    INFO:chroma:Optimization: Sufficient memory to move vertices onto GPU
    INFO:chroma:device usage:
    ----------
    nodes             2.8M  44.7M
    total                   44.7M
    ----------
    device total             2.1G
    device used            243.2M
    device free              1.9G

    /usr/local/env/chroma_env/src/chroma/chroma/gpu/tools.py:32: UserWarning: The CUDA compiler succeeded, but said the following:
    kernel.cu(198): warning: integer conversion resulted in a change of sign

    kernel.cu(198): warning: integer conversion resulted in a change of sign


      no_extern_c=True)



Black Window Issue
~~~~~~~~~~~~~~~~~~

Observe a flaky but now repeatable situation where the nodes total is overstated and 
get a black window::

    (chroma_env)delta:~ blyth$ chroma-cam $ENV_HOME/geant4/geometry/materials/g4_00.dae
    WARNING:env.geant4.geometry.collada.daenode:failed to find parent for   top.0             -  (failure expected only for root node)
    INFO:chroma:Flattening detector mesh...
    INFO:chroma:  triangles: 2483638
    INFO:chroma:  vertices:  1264041
    INFO:chroma:Loading BVH "default" for geometry from cache.
    INFO:chroma:loaded geometry <chroma.geometry.Geometry object at 0x10f9cb190> 
    INFO:chroma:starting view [1024, 576] 
    INFO:chroma:create Camera 
    INFO:chroma:Camera.__init__
    INFO:chroma:Camera.__init__ done
    INFO:chroma:_run Camera 
    INFO:chroma:Optimization: Sufficient memory to move triangles onto GPU
    INFO:chroma:Optimization: Sufficient memory to move vertices onto GPU
    INFO:chroma:device usage:
    ----------
    nodes             3.4M  54.0M
    total                   54.0M
    ----------
    device total             2.1G
    device used            318.7M
    device free              1.8G

    (chroma_env)delta:~ blyth$ 



Clearing the BVH cache, fails to resolve::

    (chroma_env)delta:~ blyth$ rm -rf  ~/.chroma/bvh
    (chroma_env)delta:~ blyth$ chroma-cam $ENV_HOME/geant4/geometry/materials/g4_00.dae
    WARNING:env.geant4.geometry.collada.daenode:failed to find parent for   top.0             -  (failure expected only for root node)
    INFO:chroma:Flattening detector mesh...
    INFO:chroma:  triangles: 2483638
    INFO:chroma:  vertices:  1264041
    INFO:chroma:Building new BVH using recursive grid algorithm.
    Expanding 20256 parent nodes
    Merging 2483638 nodes to 624241 parents
    Expanding 543 parent nodes
    Merging 651151 nodes to 167090 parents
    Expanding 223 parent nodes
    Merging 167863 nodes to 50865 parents
    Expanding 49 parent nodes
    Merging 51088 nodes to 15739 parents
    Expanding 9 parent nodes
    Merging 15788 nodes to 3680 parents
    Merging 3689 nodes to 897 parents
    Merging 897 nodes to 205 parents
    Merging 205 nodes to 51 parents
    Merging 51 nodes to 15 parents
    Merging 15 nodes to 4 parents
    Merging 4 nodes to 1 parent
    INFO:chroma:BVH generated in 3.6 seconds.
    INFO:chroma:Saving BVH (ce8a94af92438d400589a5faeb5b9f37:default) to cache.
    INFO:chroma:loaded geometry <chroma.geometry.Geometry object at 0x1151c2190> 
    INFO:chroma:starting view [1024, 576] 
    INFO:chroma:create Camera 
    INFO:chroma:Camera.__init__
    INFO:chroma:Camera.__init__ done
    INFO:chroma:_run Camera 
    INFO:chroma:Optimization: Sufficient memory to move triangles onto GPU
    INFO:chroma:Optimization: Sufficient memory to move vertices onto GPU
    INFO:chroma:device usage:
    ----------
    nodes             3.4M  54.0M
    total                   54.0M
    ----------
    device total             2.1G
    device used            493.5M
    device free              1.7G



Switch on CUDA_PROFILE in hope to learn where pycuda is caching its kernels, no joy::

    (chroma_env)delta:chroma_camera blyth$ CUDA_PROFILE=1 chroma-cam $ENV_HOME/geant4/geometry/materials/g4_00.dae
    ...

    (chroma_env)delta:chroma_camera blyth$ head -10 cuda_profile_0.log 
    # CUDA_PROFILE_LOG_VERSION 2.0
    # CUDA_DEVICE 0 GeForce GT 750M
    # CUDA_CONTEXT 1
    method,gputime,cputime,occupancy
    method=[ write_size ] gputime=[ 7.424 ] cputime=[ 17.288 ] occupancy=[ 0.016 ] 
    method=[ memcpyDtoH ] gputime=[ 5.280 ] cputime=[ 538.104 ] 
    method=[ write_size ] gputime=[ 6.848 ] cputime=[ 9.459 ] occupancy=[ 0.016 ] 
    method=[ memcpyDtoH ] gputime=[ 4.352 ] cputime=[ 1097.478 ] 
    method=[ write_size ] gputime=[ 4.256 ] cputime=[ 7.456 ] occupancy=[ 0.016 ] 
    method=[ memcpyDtoH ] gputime=[ 3.872 ] cputime=[ 17.787 ] 

    (chroma_env)delta:chroma_camera blyth$ tail -10 cuda_profile_0.log 
    method=[ memcpyHtoD ] gputime=[ 1.344 ] cputime=[ 1.983 ] 
    method=[ memcpyHtoD ] gputime=[ 1.312 ] cputime=[ 2.374 ] 
    method=[ memcpyHtoD ] gputime=[ 1.344 ] cputime=[ 1.941 ] 
    method=[ memcpyHtoD ] gputime=[ 1.344 ] cputime=[ 1.965 ] 
    method=[ memcpyHtoD ] gputime=[ 1142.528 ] cputime=[ 1021.808 ] 
    method=[ memcpyHtoD ] gputime=[ 1144.320 ] cputime=[ 989.674 ] 
    method=[ fill ] gputime=[ 51.712 ] cputime=[ 11.983 ] occupancy=[ 1.000 ] 
    method=[ fill ] gputime=[ 54.016 ] cputime=[ 16.531 ] occupancy=[ 1.000 ] 
    method=[ render ] gputime=[ 1384.512 ] cputime=[ 224.247 ] occupancy=[ 0.500 ] 
    method=[ memcpyDtoH ] gputime=[ 578.752 ] cputime=[ 2295.950 ] 


After a restart, same problem::

    delta:chroma_camera blyth$ cat chroma_camera_test.sh 
    #!/bin/bash -l

    chroma-
    which python
    which chroma-cam

    chroma-cam $ENV_HOME/geant4/geometry/materials/g4_00.dae

    delta:chroma_camera blyth$ ./chroma_camera_test.sh 
    /usr/local/env/chroma_env/bin/python
    /usr/local/env/chroma_env/bin/chroma-cam
    WARNING:env.geant4.geometry.collada.daenode:failed to find parent for   top.0             -  (failure expected only for root node)
    INFO:chroma:Flattening detector mesh...
    INFO:chroma:  triangles: 2483638
    INFO:chroma:  vertices:  1264041
    INFO:chroma:Loading BVH "default" for geometry from cache.
    INFO:chroma:loaded geometry <chroma.geometry.Geometry object at 0x110de9190> 
    INFO:chroma:starting view [1024, 576] 
    INFO:chroma:create Camera 
    INFO:chroma:Camera.__init__
    INFO:chroma:Camera.__init__ done
    INFO:chroma:_run Camera 
    /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/characterize.py:40: UserWarning: The CUDA compiler succeeded, but said the following:
    kernel.cu(45): warning: variable "NCHILD_MASK" was declared but never referenced

    kernel.cu(45): warning: variable "NCHILD_MASK" was declared but never referenced


      """ % (preamble, type_name), no_extern_c=True)
    INFO:chroma:Optimization: Sufficient memory to move triangles onto GPU
    INFO:chroma:Optimization: Sufficient memory to move vertices onto GPU
    INFO:chroma:device usage:
    ----------
    nodes             3.4M  54.0M
    total                   54.0M
    ----------
    device total             2.1G
    device used            305.1M
    device free              1.8G

    /usr/local/env/chroma_env/src/chroma/chroma/gpu/tools.py:32: UserWarning: The CUDA compiler succeeded, but said the following:
    kernel.cu(198): warning: integer conversion resulted in a change of sign

    kernel.cu(198): warning: integer conversion resulted in a change of sign


      no_extern_c=True)



Careful observation of output (triangle counts) reveals the cause to be the attempt to 
skip the overly large top.0 node, which I though I had backed out of.::

    delta:chroma_camera blyth$ ./chroma_camera_test.sh 
    /usr/local/env/chroma_env/bin/python
    /usr/local/env/chroma_env/bin/chroma-cam
    WARNING:env.geant4.geometry.collada.daenode:failed to find parent for   top.0             -  (failure expected only for root node)
    WARNING:env.geant4.geometry.collada.collada_to_chroma:setting parent_material to __dd__Materials__Vacuum0xaf1d298 as parent is None for node top.0 
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
    INFO:chroma:BVH generated in 6.4 seconds.
    INFO:chroma:Saving BVH (c600c30494ebdd3c9fed0474e09f9a80:default) to cache.
    INFO:chroma:loaded geometry <chroma.geometry.Geometry object at 0x1173d1150> 
    INFO:chroma:starting view [1024, 576] 
    INFO:chroma:create Camera 
    INFO:chroma:Camera.__init__
    INFO:chroma:Camera.__init__ done
    INFO:chroma:_run Camera 
    INFO:chroma:Optimization: Sufficient memory to move triangles onto GPU
    INFO:chroma:Optimization: Sufficient memory to move vertices onto GPU
    INFO:chroma:device usage:
    ----------
    nodes             2.8M  44.7M
    total                   44.7M
    ----------
    device total             2.1G
    device used            362.0M
    device free              1.8G

    delta:chroma_camera blyth$ 



