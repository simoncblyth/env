Unittests
=============

Initially, Feb/Mar 2014
------------------------


::

    (chroma_env)delta:chroma blyth$ which python
    /usr/local/env/chroma_env/bin/python

    (chroma_env)delta:chroma blyth$ python setup.py test
    running test
    running egg_info
    writing requirements to Chroma.egg-info/requires.txt
    writing Chroma.egg-info/PKG-INFO
    writing top-level names to Chroma.egg-info/top_level.txt
    writing dependency_links to Chroma.egg-info/dependency_links.txt
    reading manifest file 'Chroma.egg-info/SOURCES.txt'
    writing manifest file 'Chroma.egg-info/SOURCES.txt'
    running build_ext
    copying build/lib.macosx-10.9-x86_64-2.7/chroma/generator/_g4chroma.so -> chroma/generator
    copying build/lib.macosx-10.9-x86_64-2.7/chroma/generator/mute.so -> chroma/generator
    test.linalg_test.testfloat3add ... ok
    test.linalg_test.testfloat3sub ... ok
    test.linalg_test.testfloat3addequal ... ok
    test.linalg_test.testfloat3subequal ... ok
    test.linalg_test.testfloat3addfloat ... ok
    test.linalg_test.testfloat3addfloatequal ... ok
    test.linalg_test.testfloataddfloat3 ... ok
    test.linalg_test.testfloat3subfloat ... ok
    test.linalg_test.testfloat3subfloatequal ... ok
    test.linalg_test.testfloatsubfloat3 ... ok
    test.linalg_test.testfloat3mulfloat ... ok
    test.linalg_test.testfloat3mulfloatequal ... ok
    test.linalg_test.testfloatmulfloat3 ... ok
    test.linalg_test.testfloat3divfloat ... ok
    test.linalg_test.testfloat3divfloatequal ... ok
    test.linalg_test.testfloatdivfloat3 ... ok
    test.linalg_test.testdot ... ok
    test.linalg_test.testcross ... ok
    test.linalg_test.testnorm ... ok
    test.linalg_test.testminusfloat3 ... ok
    test.matrix_test.test_matrix ... ok
    test.rotate_test.test_rotate ... ok
    test_get_layer (test.test_bvh.TestBVH) ... ok
    test_layer_count (test.test_bvh.TestBVH) ... ok
    test_len (test.test_bvh.TestBVH) ... ok
    test_area (test.test_bvh.TestBVHLayer) ... ok
    test_fixed_array_to_world (test.test_bvh.TestWorldCoords) ... ok
    test_fixed_to_world (test.test_bvh.TestWorldCoords) ... ok
    test_out_of_range (test.test_bvh.TestWorldCoords) ... ok
    test_world_array_to_fixed (test.test_bvh.TestWorldCoords) ... ok
    test_world_to_fixed (test.test_bvh.TestWorldCoords) ... ok
    test.test_bvh.test_unpack_nodes ... ok
    test.test_bvh_simple.test_simple(2,) ... ok
    test.test_bvh_simple.test_simple(3,) ... ok
    test.test_bvh_simple.test_simple(4,) ... ok
    test_exist_bvh (test.test_cache.TestCacheBVH) ... ok
    test_list_bvh (test.test_cache.TestCacheBVH) ... ok
    test_load_bvh_not_found (test.test_cache.TestCacheBVH) ... ok
    test_remove_bvh (test.test_cache.TestCacheBVH) ... ok
    test_save_load_new_bvh (test.test_cache.TestCacheBVH) ... ok
    test_creation (test.test_cache.TestCacheCreation) ... ok
    test_recreation (test.test_cache.TestCacheCreation) ... ok
    test_default_geometry (test.test_cache.TestCacheGeometry) ... ok
    test_default_geometry_corruption (test.test_cache.TestCacheGeometry) ... ok
    test_get_geometry_hash (test.test_cache.TestCacheGeometry) ... ok
    test_get_geometry_hash_not_found (test.test_cache.TestCacheGeometry) ... ok
    test_list_geometry (test.test_cache.TestCacheGeometry) ... ok
    test_load_geometry_not_found (test.test_cache.TestCacheGeometry) ... ok
    test_remove_geometry (test.test_cache.TestCacheGeometry) ... ok
    test_replace_geometry (test.test_cache.TestCacheGeometry) ... ok
    test_save_load_new_geometry (test.test_cache.TestCacheGeometry) ... ok
    test_exist_dir (test.test_cache.TestVerifyOrCreateDir) ... ok
    test_exist_file (test.test_cache.TestVerifyOrCreateDir) ... ok
    test_no_dir (test.test_cache.TestVerifyOrCreateDir) ... ok
    testCharge (test.test_detector.TestDetector)
    Test PMT charge distribution ... ok
    testTime (test.test_detector.TestDetector)
    Test PMT time distribution ... FAIL
    test_center (test.test_generator_photon.TestG4ParallelGenerator)
    Generate Cherenkov light at the center of the world volume ... ok
    test_off_center (test.test_generator_photon.TestG4ParallelGenerator)
    Generate Cherenkov light at (1 m, 0 m, 0 m) ... ok
    test_constant_particle_gun_center (test.test_generator_vertex.TestParticleGun)
    Generate electron vertices at the center of the world volume. ... ok
    test_off_center (test.test_generator_vertex.TestParticleGun)
    Generate electron vertices at (1,0,0) in the world volume. ... ok
    test_file_write_and_read (test.test_io.TestRootIO) ... ok
    test_parabola_eval (test.test_parabola.Test1D) ... ok
    test_solve (test.test_parabola.Test1D) ... ok
    test_parabola_eval (test.test_parabola.Test2D) ... ok
    test_solve (test.test_parabola.Test2D) ... ok
    testGPUPDF (test.test_pdf.TestPDF)
    Create a hit count and (q,t) PDF for 10 MeV events in MicroLBNE ... ok
    testSimPDF (test.test_pdf.TestPDF) ... ok
    testAbort (test.test_propagation.TestPropagation)
    Photons that hit a triangle at normal incidence should not abort. ... ok
    test_intersection_distance (test.test_ray_intersection.TestRayIntersection) ... SKIP: Ray data file needs to be updated
    testAngularDistributionPolarized (test.test_rayleigh.TestRayleigh) ... ok
    testBulkReemission (test.test_reemission.TestReemission)
    Test bulk reemission ... SKIP: need to implement scipy stats functions here
    test_sampling (test.test_sample_cdf.TestSampling)
    Verify that the CDF-based sampler on the GPU reproduces a binned ... ok

    ======================================================================
    FAIL: testTime (test.test_detector.TestDetector)
    Test PMT time distribution
    ----------------------------------------------------------------------
    Traceback (most recent call last):
      File "/usr/local/env/chroma_env/src/chroma/test/test_detector.py", line 50, in testTime
        self.assertAlmostEqual(hit_times.std(),  1.2, delta=1e-1)
    AssertionError: 0.02522058 != 1.2 within 0.1 delta
    -------------------- >> begin captured stdout << ---------------------
    Merging 24 nodes to 8 parents
    Merging 8 nodes to 2 parents
    Merging 2 nodes to 1 parent

    --------------------- >> end captured stdout << ----------------------
    -------------------- >> begin captured logging << --------------------
    chroma: INFO: Flattening detector mesh...
    chroma: INFO:   triangles: 24
    chroma: INFO:   vertices:  10
    chroma: INFO: Building new BVH using recursive grid algorithm.
    chroma: INFO: BVH generated in 0.2 seconds.
    chroma: INFO: Optimization: Sufficient memory to move triangles onto GPU
    chroma: INFO: Optimization: Sufficient memory to move vertices onto GPU
    chroma: INFO: device usage:
    ----------
    nodes            35.0  560.0 
    total                  560.0 
    ----------
    device total             2.1G
    device used            336.0M
    device free              1.8G

    --------------------- >> end captured logging << ---------------------

    ----------------------------------------------------------------------
    Ran 72 tests in 47.072s

    FAILED (failures=1, skipped=2)

    /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/autoinit.py:16: RuntimeWarning: Parent module 'pycuda' not found while handling absolute import
      from pycuda.tools import clear_context_caches
    Error in atexit._run_exitfuncs:
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/atexit.py", line 24, in _run_exitfuncs
        func(targs, kargs)
      File "/usr/local/env/chroma_env/src/root-v5.34.14/lib/ROOT.py", line 593, in cleanup
        facade = sys.modules[ __name__ ]
    KeyError: 'ROOT'
    Error in sys.exitfunc:
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/atexit.py", line 24, in _run_exitfuncs
        func(targs, kargs)
      File "/usr/local/env/chroma_env/src/root-v5.34.14/lib/ROOT.py", line 593, in cleanup
        facade = sys.modules[ __name__ ]
    KeyError: 'ROOT'
    (chroma_env)delta:chroma blyth$ 





atexit mess avoidance
~~~~~~~~~~~~~~~~~~~~~~

Adding "import ROOT" into the setup.py avoids the atexit mess::

    Error in atexit._run_exitfuncs:
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/atexit.py", line 24, in _run_exitfuncs
        func(*targs, **kargs)
      File "/usr/local/env/chroma_env/src/root-v5.34.14/lib/ROOT.py", line 593, in cleanup
        facade = sys.modules[ __name__ ]
    KeyError: 'ROOT'
    Error in sys.exitfunc:
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/atexit.py", line 24, in _run_exitfuncs
        func(*targs, **kargs)
      File "/usr/local/env/chroma_env/src/root-v5.34.14/lib/ROOT.py", line 593, in cleanup
        facade = sys.modules[ __name__ ]
    KeyError: 'ROOT'
 

parent module 'pycuda' not found while handling absolute import
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This warning remains::

    /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/autoinit.py:16: RuntimeWarning: Parent module 'pycuda' not found while handling absolute import
      from pycuda.tools import clear_context_caches

::

    (chroma_env)delta:chroma-deps blyth$ cat /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/autoinit.py
    import pycuda.driver as cuda

    # Initialize CUDA
    cuda.init()

    from pycuda.tools import make_default_context
    global context
    context = make_default_context()
    device = context.get_device()

    def _finish_up():
        global context
        context.pop()
        context = None

        from pycuda.tools import clear_context_caches
        clear_context_caches()

    import atexit
    atexit.register(_finish_up)


::

    (chroma_env)delta:chroma blyth$ python -c "import pycuda"
    (chroma_env)delta:chroma blyth$ 
    (chroma_env)delta:chroma blyth$ python -c "import pycuda.tools"
    (chroma_env)delta:chroma blyth$ 
    (chroma_env)delta:chroma blyth$ python -c "from pycuda.tools import clear_context_caches"
    (chroma_env)delta:chroma blyth$ 



testTime Failure repeats but with different values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    ======================================================================
    FAIL: testTime (test.test_detector.TestDetector)
    Test PMT time distribution
    ----------------------------------------------------------------------
    Traceback (most recent call last):
      File "/usr/local/env/chroma_env/src/chroma/test/test_detector.py", line 50, in testTime
        self.assertAlmostEqual(hit_times.std(),  1.2, delta=1e-1)
    AssertionError: 0.02522058 != 1.2 within 0.1 delta
    -------------------- >> begin captured stdout << ---------------------
 



Doing this simulation repeatedly with a modified `test_detector.py`
suggests this fail is due to insufficient stats (only 1000 rolls of a single photon)
for the 500 or 250 time/charge hit distributions to fall within expected bounds.

Need to check distribution to be sure. 


Again April 2014
------------------

* commited fix for ROOT issue to bitbucket fork

Testing again, from my bitbucket fork::

    (chroma_env)delta:chroma blyth$ python setup.py test
    running test
    running egg_info
    writing requirements to Chroma.egg-info/requires.txt
    writing Chroma.egg-info/PKG-INFO
    writing top-level names to Chroma.egg-info/top_level.txt
    writing dependency_links to Chroma.egg-info/dependency_links.txt
    reading manifest file 'Chroma.egg-info/SOURCES.txt'
    writing manifest file 'Chroma.egg-info/SOURCES.txt'
    running build_ext
    copying build/lib.macosx-10.9-x86_64-2.7/chroma/generator/_g4chroma.so -> chroma/generator
    copying build/lib.macosx-10.9-x86_64-2.7/chroma/generator/mute.so -> chroma/generator
    Info in <TUnixSystem::ACLiC>: creating shared library /Users/blyth/.chroma/root_C.so

    RooFit v3.59 -- Developed by Wouter Verkerke and David Kirkby 
                    Copyright (C) 2000-2013 NIKHEF, University of California & Stanford University
                    All rights reserved, please read http://roofit.sourceforge.net/license.txt

    test.linalg_test.testfloat3add ... ok
    test.linalg_test.testfloat3sub ... ok
    test.linalg_test.testfloat3addequal ... ok
    test.linalg_test.testfloat3subequal ... ok
    test.linalg_test.testfloat3addfloat ... ok
    test.linalg_test.testfloat3addfloatequal ... ok
    test.linalg_test.testfloataddfloat3 ... ok
    test.linalg_test.testfloat3subfloat ... ok
    test.linalg_test.testfloat3subfloatequal ... ok
    test.linalg_test.testfloatsubfloat3 ... ok
    test.linalg_test.testfloat3mulfloat ... ok
    test.linalg_test.testfloat3mulfloatequal ... ok
    test.linalg_test.testfloatmulfloat3 ... ok
    test.linalg_test.testfloat3divfloat ... ok
    test.linalg_test.testfloat3divfloatequal ... ok
    test.linalg_test.testfloatdivfloat3 ... ok
    test.linalg_test.testdot ... ok
    test.linalg_test.testcross ... ok
    test.linalg_test.testnorm ... ok
    test.linalg_test.testminusfloat3 ... ok
    test.matrix_test.test_matrix ... ok
    test.rotate_test.test_rotate ... ok
    test_get_layer (test.test_bvh.TestBVH) ... ok
    test_layer_count (test.test_bvh.TestBVH) ... ok
    test_len (test.test_bvh.TestBVH) ... ok
    test_area (test.test_bvh.TestBVHLayer) ... ok
    test_fixed_array_to_world (test.test_bvh.TestWorldCoords) ... ok
    test_fixed_to_world (test.test_bvh.TestWorldCoords) ... ok
    test_out_of_range (test.test_bvh.TestWorldCoords) ... ok
    test_world_array_to_fixed (test.test_bvh.TestWorldCoords) ... ok
    test_world_to_fixed (test.test_bvh.TestWorldCoords) ... ok
    test.test_bvh.test_unpack_nodes ... ok
    test.test_bvh_simple.test_simple(2,) ... ok
    test.test_bvh_simple.test_simple(3,) ... ok
    test.test_bvh_simple.test_simple(4,) ... ok
    test_exist_bvh (test.test_cache.TestCacheBVH) ... ok
    test_list_bvh (test.test_cache.TestCacheBVH) ... ok
    test_load_bvh_not_found (test.test_cache.TestCacheBVH) ... ok
    test_remove_bvh (test.test_cache.TestCacheBVH) ... ok
    test_save_load_new_bvh (test.test_cache.TestCacheBVH) ... ok
    test_creation (test.test_cache.TestCacheCreation) ... ok
    test_recreation (test.test_cache.TestCacheCreation) ... ok
    test_default_geometry (test.test_cache.TestCacheGeometry) ... ok
    test_default_geometry_corruption (test.test_cache.TestCacheGeometry) ... ok
    test_get_geometry_hash (test.test_cache.TestCacheGeometry) ... ok
    test_get_geometry_hash_not_found (test.test_cache.TestCacheGeometry) ... ok
    test_list_geometry (test.test_cache.TestCacheGeometry) ... ok
    test_load_geometry_not_found (test.test_cache.TestCacheGeometry) ... ok
    test_remove_geometry (test.test_cache.TestCacheGeometry) ... ok
    test_replace_geometry (test.test_cache.TestCacheGeometry) ... ok
    test_save_load_new_geometry (test.test_cache.TestCacheGeometry) ... ok
    test_exist_dir (test.test_cache.TestVerifyOrCreateDir) ... ok
    test_exist_file (test.test_cache.TestVerifyOrCreateDir) ... ok
    test_no_dir (test.test_cache.TestVerifyOrCreateDir) ... ok
    testCharge (test.test_detector.TestDetector)
    Test PMT charge distribution ... ok
    testTime (test.test_detector.TestDetector)
    Test PMT time distribution ... ok
    test_center (test.test_generator_photon.TestG4ParallelGenerator)
    Generate Cherenkov light at the center of the world volume ... ok
    test_off_center (test.test_generator_photon.TestG4ParallelGenerator)
    Generate Cherenkov light at (1 m, 0 m, 0 m) ... ok
    test_constant_particle_gun_center (test.test_generator_vertex.TestParticleGun)
    Generate electron vertices at the center of the world volume. ... ok
    test_off_center (test.test_generator_vertex.TestParticleGun)
    Generate electron vertices at (1,0,0) in the world volume. ... ok
    test_file_write_and_read (test.test_io.TestRootIO) ... ok
    test_parabola_eval (test.test_parabola.Test1D) ... ok
    test_solve (test.test_parabola.Test1D) ... ok
    test_parabola_eval (test.test_parabola.Test2D) ... ok
    test_solve (test.test_parabola.Test2D) ... ok
    testGPUPDF (test.test_pdf.TestPDF)
    Create a hit count and (q,t) PDF for 10 MeV events in MicroLBNE ... ok
    testSimPDF (test.test_pdf.TestPDF) ... ok
    testAbort (test.test_propagation.TestPropagation)
    Photons that hit a triangle at normal incidence should not abort. ... ok
    test_intersection_distance (test.test_ray_intersection.TestRayIntersection) ... SKIP: Ray data file needs to be updated
    testAngularDistributionPolarized (test.test_rayleigh.TestRayleigh) ... ok
    testBulkReemission (test.test_reemission.TestReemission)
    Test bulk reemission ... SKIP: need to implement scipy stats functions here
    test_sampling (test.test_sample_cdf.TestSampling)
    Verify that the CDF-based sampler on the GPU reproduces a binned ... ok

    ----------------------------------------------------------------------
    Ran 72 tests in 81.926s

    OK (skipped=2)
    /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/autoinit.py:16: RuntimeWarning: Parent module 'pycuda' not found while handling absolute import
      from pycuda.tools import clear_context_caches
    Error in atexit._run_exitfuncs:
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/atexit.py", line 24, in _run_exitfuncs
        func(targs, kargs)
      File "/usr/local/env/chroma_env/src/root-v5.34.14/lib/ROOT.py", line 593, in cleanup
        facade = sys.modules[ __name__ ]
    KeyError: 'ROOT'
    Error in sys.exitfunc:
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/atexit.py", line 24, in _run_exitfuncs
        func(targs, kargs)
      File "/usr/local/env/chroma_env/src/root-v5.34.14/lib/ROOT.py", line 593, in cleanup
        facade = sys.modules[ __name__ ]
    KeyError: 'ROOT'
    (chroma_env)delta:chroma blyth$ 



nosetests : a nicer way to run them
---------------------------------------


::

    (chroma_env)delta:chroma blyth$ nosetests -v
    Info in <TUnixSystem::ACLiC>: creating shared library /Users/blyth/.chroma/root_C.so

    RooFit v3.59 -- Developed by Wouter Verkerke and David Kirkby 
                    Copyright (C) 2000-2013 NIKHEF, University of California & Stanford University
                    All rights reserved, please read http://roofit.sourceforge.net/license.txt

    test.linalg_test.testfloat3add ... ok
    test.linalg_test.testfloat3sub ... ok
    test.linalg_test.testfloat3addequal ... ok
    test.linalg_test.testfloat3subequal ... ok
    test.linalg_test.testfloat3addfloat ... ok
    test.linalg_test.testfloat3addfloatequal ... ok
    test.linalg_test.testfloataddfloat3 ... ok
    test.linalg_test.testfloat3subfloat ... ok
    test.linalg_test.testfloat3subfloatequal ... ok
    test.linalg_test.testfloatsubfloat3 ... ok
    test.linalg_test.testfloat3mulfloat ... ok
    test.linalg_test.testfloat3mulfloatequal ... ok
    test.linalg_test.testfloatmulfloat3 ... ok
    test.linalg_test.testfloat3divfloat ... ok
    test.linalg_test.testfloat3divfloatequal ... ok
    test.linalg_test.testfloatdivfloat3 ... ok
    test.linalg_test.testdot ... ok
    test.linalg_test.testcross ... ok
    test.linalg_test.testnorm ... ok
    test.linalg_test.testminusfloat3 ... ok
    test.matrix_test.test_matrix ... ok
    test.rotate_test.test_rotate ... ok
    test_get_layer (test.test_bvh.TestBVH) ... ok
    test_layer_count (test.test_bvh.TestBVH) ... ok
    test_len (test.test_bvh.TestBVH) ... ok
    test_area (test.test_bvh.TestBVHLayer) ... ok
    test_fixed_array_to_world (test.test_bvh.TestWorldCoords) ... ok
    test_fixed_to_world (test.test_bvh.TestWorldCoords) ... ok
    test_out_of_range (test.test_bvh.TestWorldCoords) ... ok
    test_world_array_to_fixed (test.test_bvh.TestWorldCoords) ... ok
    test_world_to_fixed (test.test_bvh.TestWorldCoords) ... ok
    test.test_bvh.test_unpack_nodes ... ok
    test.test_bvh_simple.test_simple(2,) ... ok
    test.test_bvh_simple.test_simple(3,) ... ok
    test.test_bvh_simple.test_simple(4,) ... ok
    test_exist_bvh (test.test_cache.TestCacheBVH) ... ok
    test_list_bvh (test.test_cache.TestCacheBVH) ... ok
    test_load_bvh_not_found (test.test_cache.TestCacheBVH) ... ok
    test_remove_bvh (test.test_cache.TestCacheBVH) ... ok
    test_save_load_new_bvh (test.test_cache.TestCacheBVH) ... ok
    test_creation (test.test_cache.TestCacheCreation) ... ok
    test_recreation (test.test_cache.TestCacheCreation) ... ok
    test_default_geometry (test.test_cache.TestCacheGeometry) ... ok
    test_default_geometry_corruption (test.test_cache.TestCacheGeometry) ... ok
    test_get_geometry_hash (test.test_cache.TestCacheGeometry) ... ok
    test_get_geometry_hash_not_found (test.test_cache.TestCacheGeometry) ... ok
    test_list_geometry (test.test_cache.TestCacheGeometry) ... ok
    test_load_geometry_not_found (test.test_cache.TestCacheGeometry) ... ok
    test_remove_geometry (test.test_cache.TestCacheGeometry) ... ok
    test_replace_geometry (test.test_cache.TestCacheGeometry) ... ok
    test_save_load_new_geometry (test.test_cache.TestCacheGeometry) ... ok
    test_exist_dir (test.test_cache.TestVerifyOrCreateDir) ... ok
    test_exist_file (test.test_cache.TestVerifyOrCreateDir) ... ok
    test_no_dir (test.test_cache.TestVerifyOrCreateDir) ... ok
    Test PMT charge distribution ... ok
    Test PMT time distribution ... ok
    Generate Cherenkov light at the center of the world volume ... ok
    Generate Cherenkov light at (1 m, 0 m, 0 m) ... ok
    Generate electron vertices at the center of the world volume. ... ok
    Generate electron vertices at (1,0,0) in the world volume. ... ok
    test_file_write_and_read (test.test_io.TestRootIO) ... ok
    test_parabola_eval (test.test_parabola.Test1D) ... ok
    test_solve (test.test_parabola.Test1D) ... ok
    test_parabola_eval (test.test_parabola.Test2D) ... ok
    test_solve (test.test_parabola.Test2D) ... ok
    Create a hit count and (q,t) PDF for 10 MeV events in MicroLBNE ... ok
    testSimPDF (test.test_pdf.TestPDF) ... ok
    Photons that hit a triangle at normal incidence should not abort. ... ok
    test_intersection_distance (test.test_ray_intersection.TestRayIntersection) ... SKIP: Ray data file needs to be updated
    testAngularDistributionPolarized (test.test_rayleigh.TestRayleigh) ... ok
    Test bulk reemission ... SKIP: need to implement scipy stats functions here
    Verify that the CDF-based sampler on the GPU reproduces a binned ... ok

    ----------------------------------------------------------------------
    Ran 72 tests in 49.408s

    OK (SKIP=2)
    (chroma_env)delta:chroma blyth$ 


