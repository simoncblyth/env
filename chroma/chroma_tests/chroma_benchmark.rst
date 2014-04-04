chroma benchmark
==================

.. contents:: :local:


gzip issue
-----------

::

    (chroma_env)delta:chroma blyth$ python benchmark.py 
    Traceback (most recent call last):
      File "benchmark.py", line 10, in <module>
        from chroma import gpu
      File "/usr/local/env/chroma_env/src/chroma/chroma/__init__.py", line 2, in <module>
        from chroma.camera import Camera, EventViewer, view, build
      File "/usr/local/env/chroma_env/src/chroma/chroma/camera.py", line 15, in <module>
        from chroma.geometry import Mesh, Solid, Geometry, vacuum
      File "/usr/local/env/chroma_env/src/chroma/chroma/geometry.py", line 6, in <module>
        import gzip
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/gzip.py", line 36, in <module>
        class GzipFile(io.BufferedIOBase):
    AttributeError: 'module' object has no attribute 'BufferedIOBase'
    (chroma_env)delta:chroma blyth$ 


io module shadowing
----------------------

Module implemented `chroma/io/__init__.py`  is problematic for gzip import::

    (chroma_env)delta:chroma blyth$ 
    (chroma_env)delta:chroma blyth$ python -c "from chroma.geometry import Mesh, Solid, Geometry, vacuum"
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/usr/local/env/chroma_env/src/chroma/chroma/__init__.py", line 2, in <module>
        from chroma.camera import Camera, EventViewer, view, build
      File "/usr/local/env/chroma_env/src/chroma/chroma/camera.py", line 16, in <module>
        from chroma.geometry import Mesh, Solid, Geometry, vacuum
      File "/usr/local/env/chroma_env/src/chroma/chroma/geometry.py", line 6, in <module>
        import gzip
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/gzip.py", line 36, in <module>
        class GzipFile(io.BufferedIOBase):
    AttributeError: 'module' object has no attribute 'BufferedIOBase'
    (chroma_env)delta:chroma blyth$ 
    (chroma_env)delta:chroma blyth$ pwd
    /usr/local/env/chroma_env/src/chroma/chroma
    (chroma_env)delta:chroma blyth$ 
    (chroma_env)delta:chroma blyth$ 
    (chroma_env)delta:chroma blyth$ cd /tmp
    (chroma_env)delta:tmp blyth$ python -c "from chroma.geometry import Mesh, Solid, Geometry, vacuum"
    (chroma_env)delta:tmp blyth$ 


Rename to `io_` and change sole usage in `chroma/camera.py`::

    670 class EventViewer(Camera):
    671     # Constants for display_mode
    672     CHARGE = 0
    673     TIME = 1
    674     HIT = 2
    675 
    676     def __init__(self, geometry, filename, **kwargs):
    677         Camera.__init__(self, geometry, **kwargs)
    678         # This is really slow, so we do it here in the constructor to 
    679         # avoid slowing down the import of this module
    680         from chroma.io_.root import RootReader
    681         self.rr = RootReader(filename)
    682         self.display_mode = EventViewer.CHARGE
    683         self.sum_mode = False
    684         self.photon_display_iter = itertools.cycle(['beg','end'])
    685         self.photon_display_mode = self.photon_display_iter.next()
    686 


hg rename did not delete the io folder 
---------------------------------------

Due to the pesky pyc::

    (chroma_env)delta:chroma blyth$ l io/
    total 32
    -rw-r--r--  1 blyth  staff  9073 Apr  4 13:13 root.pyc
    -rw-r--r--  1 blyth  staff   145 Apr  4 13:13 __init__.pyc
    (chroma_env)delta:chroma blyth$ 

So still shadowing::

    (chroma_env)delta:chroma blyth$ python -c "import gzip"
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/gzip.py", line 36, in <module>
        class GzipFile(io.BufferedIOBase):
    AttributeError: 'module' object has no attribute 'BufferedIOBase'
    (chroma_env)delta:chroma blyth$ 

Until manually remove::

    (chroma_env)delta:chroma blyth$ rm -rf io
    (chroma_env)delta:chroma blyth$ python -c "import gzip"
    (chroma_env)delta:chroma blyth$ 



cuMemAlloc failed with `demo.detector()`
----------------------------------------------

::

    MemoryError: cuMemAlloc failed: out of memory


::

    (chroma_env)delta:chroma blyth$ python benchmark.py 
    Expanding 72673 parent nodes
    Merging 59444320 nodes to 18942711 parents
    Expanding 80 parent nodes
    Merging 19034296 nodes to 5999243 parents
    Merging 5999323 nodes to 1658271 parents
    Merging 1658271 nodes to 482893 parents
    Expanding 1181 parent nodes
    Merging 482893 nodes to 120973 parents
    Merging 122154 nodes to 31993 parents
    Merging 31993 nodes to 6305 parents
    Merging 6305 nodes to 1341 parents
    Merging 1341 nodes to 272 parents
    Merging 272 nodes to 56 parents
    Merging 56 nodes to 16 parents
    Merging 16 nodes to 4 parents
    Merging 4 nodes to 1 parent
    Traceback (most recent call last):
      File "benchmark.py", line 247, in <module>
        detector = create_geometry_from_obj(demo.detector())
      File "/usr/local/env/chroma_env/src/chroma/chroma/loader.py", line 189, in create_geometry_from_obj
        cuda_device=cuda_device)
      File "/usr/local/env/chroma_env/src/chroma/chroma/loader.py", line 151, in load_bvh
        bvh = make_recursive_grid_bvh(geometry.mesh, target_degree=3)
      File "/usr/local/env/chroma_env/src/chroma/chroma/bvh/grid.py", line 91, in make_recursive_grid_bvh
        nodes, layer_bounds = concatenate_layers(layers)
      File "/usr/local/env/chroma_env/src/chroma/chroma/gpu/bvh.py", line 266, in concatenate_layers
        grid=(nblocks_this_iter,1))
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py", line 355, in function_call
        handlers, arg_buf = _build_arg_buf(args)
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py", line 125, in _build_arg_buf
        arg_data.append(int(arg.get_device_alloc()))
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py", line 59, in get_device_alloc
        self.dev_alloc = mem_alloc_like(self.array)
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py", line 608, in mem_alloc_like
        return mem_alloc(ary.nbytes)
    MemoryError: cuMemAlloc failed: out of memory

    > /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py(608)mem_alloc_like()
    -> return mem_alloc(ary.nbytes)
    (Pdb) p ary
    array([(1057635393L, 1093418857L, 682436037L, 26470L),
           (1067793635L, 1063992748L, 696133269L, 26670L),
           (1023752832L, 949171942L, 825044518L, 27472L), ...,
           (3258629873L, 3428829884L, 3524907020L, 11971L),
           (3271081900L, 3401500949L, 3540373746L, 12171L),
           (3283796054L, 3260137618L, 3647395662L, 13170L)], 
          dtype=[('x', '<u4'), ('y', '<u4'), ('z', '<u4'), ('w', '<u4')])
    (Pdb) p ary.nbytes
    951109120
    (Pdb) p ary.nbytes/1E6
    951.10912
    (Pdb) 
    (Pdb) list
    603         return result
    604     
    605     # }}}
    606     
    607     def mem_alloc_like(ary):
    608  ->     return mem_alloc(ary.nbytes)
    609     
    610     # {{{ array handling
    611     
    612     def dtype_to_array_format(dtype):
    613         if dtype == np.uint8:
    (Pdb) bt
      /usr/local/env/chroma_env/src/chroma/chroma/benchmark.py(247)<module>()
    -> detector = create_geometry_from_obj(demo.detector())
      /usr/local/env/chroma_env/src/chroma/chroma/loader.py(189)create_geometry_from_obj()
    -> cuda_device=cuda_device)
      /usr/local/env/chroma_env/src/chroma/chroma/loader.py(151)load_bvh()
    -> bvh = make_recursive_grid_bvh(geometry.mesh, target_degree=3)
      /usr/local/env/chroma_env/src/chroma/chroma/bvh/grid.py(91)make_recursive_grid_bvh()
    -> nodes, layer_bounds = concatenate_layers(layers)
      /usr/local/env/chroma_env/src/chroma/chroma/gpu/bvh.py(266)concatenate_layers()
    -> grid=(nblocks_this_iter,1))
      /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py(355)function_call()
    -> handlers, arg_buf = _build_arg_buf(args)
      /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py(125)_build_arg_buf()
    -> arg_data.append(int(arg.get_device_alloc()))
      /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py(59)get_device_alloc()
    -> self.dev_alloc = mem_alloc_like(self.array)
    > /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/driver.py(608)mem_alloc_like()
    -> return mem_alloc(ary.nbytes)
    (Pdb) 




with `demo.tiny()` detector
----------------------------

Use a smaller detector to benchmark.

::

    (chroma_env)delta:chroma blyth$ python benchmark.py 
    Expanding 5701 parent nodes
    Merging 392512 nodes to 109652 parents
    Expanding 1011 parent nodes
    Merging 116190 nodes to 29325 parents
    Expanding 1 parent nodes
    Merging 30428 nodes to 8372 parents
    Merging 8373 nodes to 2304 parents
    Merging 2304 nodes to 558 parents
    Merging 558 nodes to 110 parents
    Merging 110 nodes to 32 parents
    Merging 32 nodes to 8 parents
    Merging 8 nodes to 2 parents
    Merging 2 nodes to 1 parent
    [ . . . . . . . . . . ]

    benchmark.py:48: UserWarning: Obsolete: either use ufloat(nominal_value, std_dev), ufloat(nominal_value, std_dev, tag), or the ufloat_fromstr() function, for string representations. Code can be automatically updated with python -m uncertainties.1to2 -w ProgramDirectory.
      return nphotons/ufloat((np.mean(run_times),np.std(run_times)))
    /usr/local/env/chroma_env/src/chroma/chroma/tools.py:19: UserWarning: Obsolete: the std_dev attribute should not be called anymore: use .std_dev instead of .std_dev(). Code can be automatically updated with python -m uncertainties.1to2 -w ProgramDirectory.
      msd = -int(math.floor(math.log10(x.std_dev())))
    /usr/local/env/chroma_env/src/chroma/chroma/tools.py:21: UserWarning: Obsolete: the std_dev attribute should not be called anymore: use .std_dev instead of .std_dev(). Code can be automatically updated with python -m uncertainties.1to2 -w ProgramDirectory.
      msd, round(x.std_dev(), msd))

    5000000 +/- 1000000 ray intersections/sec.
    [ . . . . . . . . . . ]

    benchmark.py:70: UserWarning: Obsolete: either use ufloat(nominal_value, std_dev), ufloat(nominal_value, std_dev, tag), or the ufloat_fromstr() function, for string representations. Code can be automatically updated with python -m uncertainties.1to2 -w ProgramDirectory.
      return nphotons/ufloat((np.mean(run_times),np.std(run_times)))

    32000000 +/- 2000000 photons loaded/sec.
    [ . . . . . . . . . . ]

    benchmark.py:98: UserWarning: Obsolete: either use ufloat(nominal_value, std_dev), ufloat(nominal_value, std_dev, tag), or the ufloat_fromstr() function, for string representations. Code can be automatically updated with python -m uncertainties.1to2 -w ProgramDirectory.
      return nphotons/ufloat((np.mean(run_times),np.std(run_times)))

    2440000 +/- 50000 photons propagated/sec.
    [ . . . . . . . . . . ]

    benchmark.py:156: UserWarning: Obsolete: either use ufloat(nominal_value, std_dev), ufloat(nominal_value, std_dev, tag), or the ufloat_fromstr() function, for string representations. Code can be automatically updated with python -m uncertainties.1to2 -w ProgramDirectory.
      return nevents*nreps*ndaq/ufloat((np.mean(run_times),np.std(run_times)))

    79 +/- 2 100 MeV events histogrammed/s

    Traceback (most recent call last):
      File "benchmark.py", line 278, in <module>
        tools.ufloat_to_str(pdf_eval(gpu_detector))
      File "benchmark.py", line 190, in pdf_eval
        gpu_daq.acquire(gpu_photons, rng_states, nthreads_per_block, max_blocks).get()
    AttributeError: 'NoneType' object has no attribute 'get'

    > /usr/local/env/chroma_env/src/chroma/chroma/benchmark.py(190)pdf_eval()
    -> gpu_daq.acquire(gpu_photons, rng_states, nthreads_per_block, max_blocks).get()
    (Pdb) 

    (Pdb) p gpu_daq
    <chroma.gpu.daq.GPUDaq object at 0x10bc05ed0>

    (Pdb) p gpu_photons
    <chroma.gpu.photon.GPUPhotons object at 0x10bbd1710>

    (Pdb) p rng_states
    <pycuda._driver.DeviceAllocation object at 0x10ca47fa0>

    (Pdb) p nthreads_per_block
    64
    (Pdb) p max_blocks
    1024
    (Pdb) p gpu_detector
    <chroma.gpu.detector.GPUDetector object at 0x10b5ce810>
    (Pdb) p context
    <pycuda._driver.Context object at 0x10b5cf758>
    (Pdb) 


Update for changed API
--------------------------

::

    (chroma_env)delta:chroma blyth$ python benchmark.py 
    [ . . . . . . . . . . ]
    benchmark.py:48: UserWarning: Obsolete: either use ufloat(nominal_value, std_dev), ufloat(nominal_value, std_dev, tag), or the ufloat_fromstr() function, for string representations. Code can be automatically updated with python -m uncertainties.1to2 -w ProgramDirectory.
      return nphotons/ufloat((np.mean(run_times),np.std(run_times)))
    /usr/local/env/chroma_env/src/chroma/chroma/tools.py:19: UserWarning: Obsolete: the std_dev attribute should not be called anymore: use .std_dev instead of .std_dev(). Code can be automatically updated with python -m uncertainties.1to2 -w ProgramDirectory.
      msd = -int(math.floor(math.log10(x.std_dev())))
    /usr/local/env/chroma_env/src/chroma/chroma/tools.py:21: UserWarning: Obsolete: the std_dev attribute should not be called anymore: use .std_dev instead of .std_dev(). Code can be automatically updated with python -m uncertainties.1to2 -w ProgramDirectory.
      msd, round(x.std_dev(), msd))
    5000000 +/- 1000000 ray intersections/sec.
    [ . . . . . . . . . . ]
    benchmark.py:70: UserWarning: Obsolete: either use ufloat(nominal_value, std_dev), ufloat(nominal_value, std_dev, tag), or the ufloat_fromstr() function, for string representations. Code can be automatically updated with python -m uncertainties.1to2 -w ProgramDirectory.
      return nphotons/ufloat((np.mean(run_times),np.std(run_times)))
    20000000 +/- 4000000 photons loaded/sec.
    [ . . . . . . . . . . ]
    benchmark.py:98: UserWarning: Obsolete: either use ufloat(nominal_value, std_dev), ufloat(nominal_value, std_dev, tag), or the ufloat_fromstr() function, for string representations. Code can be automatically updated with python -m uncertainties.1to2 -w ProgramDirectory.
      return nphotons/ufloat((np.mean(run_times),np.std(run_times)))
    2200000 +/- 300000 photons propagated/sec.
    [ . . . . . . . . . . ]
    benchmark.py:156: UserWarning: Obsolete: either use ufloat(nominal_value, std_dev), ufloat(nominal_value, std_dev, tag), or the ufloat_fromstr() function, for string representations. Code can be automatically updated with python -m uncertainties.1to2 -w ProgramDirectory.
      return nevents*nreps*ndaq/ufloat((np.mean(run_times),np.std(run_times)))
    78 +/- 3 100 MeV events histogrammed/s
    [ . . . . . . . . . . ]
    benchmark.py:241: UserWarning: Obsolete: either use ufloat(nominal_value, std_dev), ufloat(nominal_value, std_dev, tag), or the ufloat_fromstr() function, for string representations. Code can be automatically updated with python -m uncertainties.1to2 -w ProgramDirectory.
      return nevents*nreps*ndaq/ufloat((np.mean(run_times),np.std(run_times)))
    8500 +/- 600 100 MeV events/s accumulated in PDF evaluation data structure (100 GEANT4 x 16 Chroma x 128 DAQ)
    (chroma_env)delta:chroma blyth$ 



After uncertainties refactor
-----------------------------

::

    (chroma_env)delta:chroma blyth$ python benchmark.py 
    [ . . . . . . . . . . ]
    5000000 +/- 1000000 ray intersections/sec.
    [ . . . . . . . . . . ]
    18400000 +/- 600000 photons loaded/sec.
    [ . . . . . . . . . . ]
    2100000 +/- 200000 photons propagated/sec.
    [ . . . . . . . . . . ]
    77 +/- 5 100 MeV events histogrammed/s
    [ . . . . . . . . . . ]
    8800 +/- 400 100 MeV events/s accumulated in PDF evaluation data structure (100 GEANT4 x 16 Chroma x 128 DAQ)
    (chroma_env)delta:chroma blyth$ 


