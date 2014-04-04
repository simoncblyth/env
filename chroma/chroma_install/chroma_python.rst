Python issues
================


Locale
-------

On Mavericks 10.9, whilst trying to find how to run chroma tests::

    (chroma_env)delta:chroma blyth$ python setup.py --help-commands
    Traceback (most recent call last):
      File "setup.py", line 88, in <module>
        test_suite = 'nose.collector',
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/distutils/core.py", line 138, in setup
        ok = dist.parse_command_line()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/dist.py", line 250, in parse_command_line
        result = _Distribution.parse_command_line(self)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/distutils/dist.py", line 464, in parse_command_line
        if self.handle_display_options(option_order):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/dist.py", line 611, in handle_display_options
        return _Distribution.handle_display_options(self, option_order)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/distutils/dist.py", line 669, in handle_display_options
        self.print_commands()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/dist.py", line 371, in print_commands
        cmdclass = ep.load(False) # don't require extras, we're not running
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pkg_resources.py", line 2029, in load
        entry = __import__(self.module_name, globals(),globals(), ['__name__'])
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/Sphinx-1.2-py2.7.egg/sphinx/setup_command.py", line 20, in <module>
        from sphinx.application import Sphinx
      File "/usr/local/env/chroma_env/lib/python2.7/site-packages/Sphinx-1.2-py2.7.egg/sphinx/application.py", line 22, in <module>
        from docutils.parsers.rst import convert_directive_function, \
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/docutils/parsers/rst/__init__.py", line 74, in <module>
        import docutils.statemachine
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/docutils/statemachine.py", line 113, in <module>
        from docutils import utils
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/docutils/utils/__init__.py", line 20, in <module>
        import docutils.io
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/docutils/io.py", line 18, in <module>
        from docutils.utils.error_reporting import locale_encoding, ErrorString, ErrorOutput
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/docutils/utils/error_reporting.py", line 47, in <module>
        locale_encoding = locale.getlocale()[1] or locale.getdefaultlocale()[1]
      File "/usr/local/env/chroma_env/lib/python2.7/locale.py", line 511, in getdefaultlocale
        return _parse_localename(localename)
      File "/usr/local/env/chroma_env/lib/python2.7/locale.py", line 443, in _parse_localename
        raise ValueError, 'unknown locale: %s' % localename
    ValueError: unknown locale: UTF-8
    (chroma_env)delta:chroma blyth$ 
    (chroma_env)delta:chroma blyth$ 


Perhaps resolved by adding to .bash_profile::

    # http://stackoverflow.com/questions/19961239/pelican-3-3-pelican-quickstart-error-valueerror-unknown-locale-utf-8
    export LC_ALL=en_US.UTF-8
    export LANG=en_US.UTF-8


help commands trys to install stuff
--------------------------------------

Due to a wrong python mixup, despite the prompt suggesting inside virtualenv::

    (chroma_env)delta:chroma blyth$ python setup.py --help-commands
    Downloading http://pypi.python.org/packages/source/d/distribute/distribute-0.6.35.tar.gz
    Extracting in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-BTruBO/PyUblas-2013.1/temp/tmpWmfpCf
    Now working in /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-BTruBO/PyUblas-2013.1/temp/tmpWmfpCf/distribute-0.6.35
    Building a Distribute egg in /private/var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-BTruBO/PyUblas-2013.1
    /private/var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-BTruBO/PyUblas-2013.1/distribute-0.6.35-py2.7.egg
    -------------------------------------------------------------------------
    Setuptools conflict detected.
    -------------------------------------------------------------------------
    When I imported setuptools, I did not get the distribute version of
    setuptools, which is troubling--this package really wants to be used
    with distribute rather than the old setuptools package. More than likely,
    you have both distribute and setuptools installed, which is bad.

    See this page for more information:
    http://wiki.tiker.net/DistributeVsSetuptools

    ...


python version mixup
----------------------

Somehow despite the prompt was hooked up to macports python
rather than the virtualenv one.

::

    (chroma_env)delta:chroma blyth$ which python
    /opt/local/bin/python
    (chroma_env)delta:chroma blyth$ deactivate
    delta:chroma blyth$ 
    delta:chroma blyth$ 
    delta:chroma blyth$ chroma-
    (chroma_env)delta:chroma blyth$ which python
    /usr/local/env/chroma_env/bin/python
    (chroma_env)delta:chroma blyth$


test run
-----------


::
 
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
    AssertionError: 3.0949438 != 1.2 within 0.1 delta
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
    device used            316.0M
    device free              1.8G

    --------------------- >> end captured logging << ---------------------

    ----------------------------------------------------------------------
    Ran 72 tests in 54.758s

    FAILED (failures=1, skipped=2)
    /usr/local/env/chroma_env/lib/python2.7/site-packages/pycuda/autoinit.py:16: RuntimeWarning: Parent module 'pycuda' not found while handling absolute import
      from pycuda.tools import clear_context_caches
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
    (chroma_env)delta:chroma blyth$ 




The failure repeats but with different numbers::

    ======================================================================
    FAIL: testTime (test.test_detector.TestDetector)
    Test PMT time distribution
    ----------------------------------------------------------------------
    Traceback (most recent call last):
      File "/usr/local/env/chroma_env/src/chroma/test/test_detector.py", line 50, in testTime
        self.assertAlmostEqual(hit_times.std(),  1.2, delta=1e-1)
    AssertionError: 3.0949438 != 1.2 within 0.1 delta
    -------------------- >> begin captured stdout << ---------------------
 

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



