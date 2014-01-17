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

::

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
    -------------------------------------------------------------------------
    I will continue after a short while, fingers crossed.
    Hit Ctrl-C now if you'd like to think about the situation.
    -------------------------------------------------------------------------
    ^CTraceback (most recent call last):
      File "setup.py", line 88, in <module>
        test_suite = 'nose.collector',
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/distutils/core.py", line 112, in setup
        _setup_distribution = dist = klass(attrs)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/dist.py", line 239, in __init__
        self.fetch_build_eggs(attrs.pop('setup_requires'))
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/dist.py", line 263, in fetch_build_eggs
        parse_requirements(requires), installer=self.fetch_build_egg
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pkg_resources.py", line 564, in resolve
        dist = best[req.key] = env.best_match(req, self, installer)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pkg_resources.py", line 802, in best_match
        return self.obtain(req, installer) # try and download/install
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pkg_resources.py", line 814, in obtain
        return installer(requirement)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/dist.py", line 313, in fetch_build_egg
        return cmd.easy_install(req)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/command/easy_install.py", line 593, in easy_install
        return self.install_item(spec, dist.location, tmpdir, deps)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/command/easy_install.py", line 623, in install_item
        dists = self.install_eggs(spec, download, tmpdir)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/command/easy_install.py", line 809, in install_eggs
        return self.build_and_install(setup_script, setup_base)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/command/easy_install.py", line 1015, in build_and_install
        self.run_setup(setup_script, setup_base, args)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/command/easy_install.py", line 1000, in run_setup
        run_setup(setup_script, args)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/sandbox.py", line 50, in run_setup
        lambda: execfile(
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/sandbox.py", line 100, in run
        return func()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/sandbox.py", line 52, in <lambda>
        {'__file__':setup_script, '__name__':'__main__'}
      File "setup.py", line 143, in <module>
        
      File "setup.py", line 29, in main
        geant4_libs = check_output(['geant4-config','--libs']).split()
      File "/var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-BTruBO/PyUblas-2013.1/aksetup_helper.py", line 37, in <module>
      File "/var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-BTruBO/PyUblas-2013.1/aksetup_helper.py", line 15, in count_down_delay
    KeyboardInterrupt
    (chroma_env)delta:chroma blyth


ordinary setuptools unaware of shrinkwrap installs ?
------------------------------------------------------

So unable to run tests.

::

    (chroma_env)delta:chroma blyth$ python setup.py test
    running test
    Searching for unittest2
    Reading https://pypi.python.org/simple/unittest2/
    Best match: unittest2 0.5.1
    Downloading https://pypi.python.org/packages/source/u/unittest2/unittest2-0.5.1.zip#md5=1527fb89e38343945af1166342d851ee
    Processing unittest2-0.5.1.zip
    Writing /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-oyyUCY/unittest2-0.5.1/setup.cfg
    Running unittest2-0.5.1/setup.py -q bdist_egg --dist-dir /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-oyyUCY/unittest2-0.5.1/egg-dist-tmp-x_UUOT
    zip_safe flag not set; analyzing archive contents...
    unittest2.collector: module references __file__
    unittest2.loader: module references __file__
    unittest2.test.test_discovery: module references __file__

    Installed /usr/local/env/chroma_env/src/chroma/unittest2-0.5.1-py2.7.egg
    Searching for sphinx
    Reading https://pypi.python.org/simple/sphinx/
    Best match: Sphinx 1.2
    Downloading https://pypi.python.org/packages/2.7/S/Sphinx/Sphinx-1.2-py2.7.egg#md5=baa2dccdd2836ecbe160f9efd21804eb
    Processing Sphinx-1.2-py2.7.egg
    creating /usr/local/env/chroma_env/src/chroma/Sphinx-1.2-py2.7.egg
    Extracting Sphinx-1.2-py2.7.egg to /usr/local/env/chroma_env/src/chroma

    Installed /usr/local/env/chroma_env/src/chroma/Sphinx-1.2-py2.7.egg
    Searching for pycuda
    Reading https://pypi.python.org/simple/pycuda/
    Best match: pycuda 2013.1.1
    Downloading https://pypi.python.org/packages/source/p/pycuda/pycuda-2013.1.1.tar.gz#md5=acf9319ab2970d9700ed6486aa87b708
    Processing pycuda-2013.1.1.tar.gz
    Writing /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-EHju9f/pycuda-2013.1.1/setup.cfg
    Running pycuda-2013.1.1/setup.py -q bdist_egg --dist-dir /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-EHju9f/pycuda-2013.1.1/egg-dist-tmp-6LyPm9
    warning: no files found matching '*.cpp' under directory 'bpl-subset/bpl_subset/boost'
    warning: no files found matching '*.html' under directory 'bpl-subset/bpl_subset/boost'
    warning: no files found matching '*.inl' under directory 'bpl-subset/bpl_subset/boost'
    warning: no files found matching '*.txt' under directory 'bpl-subset/bpl_subset/boost'
    warning: no files found matching '*.h' under directory 'bpl-subset/bpl_subset/libs'
    warning: no files found matching '*.ipp' under directory 'bpl-subset/bpl_subset/libs'
    warning: no files found matching '*.pl' under directory 'bpl-subset/bpl_subset/libs'
    In file included from src/wrapper/wrap_curand.cpp:4:
    In file included from src/wrapper/tools.hpp:10:
    In file included from src/wrapper/numpy_init.hpp:6:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/arrayobject.h:4:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/ndarrayobject.h:17:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/ndarraytypes.h:1760:
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: "Using deprecated NumPy API, disable it by "          "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-W#warnings]
#warning "Using deprecated NumPy API, disable it by " \
     ^
    In file included from src/wrapper/wrap_curand.cpp:2:
    In file included from src/cpp/curand.hpp:6:
    In file included from /Developer/NVIDIA/CUDA-5.5/include/curand.h:59:
    /Developer/NVIDIA/CUDA-5.5/include/cuda_runtime.h:225:33: warning: function 'cudaMallocHost' is not needed and will not be emitted [-Wunneeded-internal-declaration]
    __inline__ __host__ cudaError_t cudaMallocHost(
                                    ^
    2 warnings generated.
    In file included from src/wrapper/wrap_cudadrv.cpp:7:
    In file included from src/wrapper/tools.hpp:10:
    In file included from src/wrapper/numpy_init.hpp:6:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/arrayobject.h:4:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/ndarrayobject.h:17:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/ndarraytypes.h:1760:
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: "Using deprecated NumPy API, disable it by "          "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-W#warnings]
    #warning "Using deprecated NumPy API, disable it by " \
    src/wrapper/wrap_cudadrv.cpp:508:5: warning: conversion from string literal to 'char *' is deprecated [-Wdeprecated-writable-strings]
        DECLARE_EXC(Error, NULL);
        ^
    src/wrapper/wrap_cudadrv.cpp:504:48: note: expanded from macro 'DECLARE_EXC'
      Cuda##NAME = py::handle<>(PyErr_NewException("pycuda._driver." #NAME, BASE, NULL)); \
                                                   ^
    src/wrapper/wrap_cudadrv.cpp:509:5: warning: conversion from string literal to 'char *' is deprecated [-Wdeprecated-writable-strings]
        DECLARE_EXC(MemoryError, CudaError.get());
        ^
    src/wrapper/wrap_cudadrv.cpp:504:48: note: expanded from macro 'DECLARE_EXC'
      Cuda##NAME = py::handle<>(PyErr_NewException("pycuda._driver." #NAME, BASE, NULL)); \
                                                   ^
    src/wrapper/wrap_cudadrv.cpp:510:5: warning: conversion from string literal to 'char *' is deprecated [-Wdeprecated-writable-strings]
        DECLARE_EXC(LogicError, CudaError.get());
        ^
    src/wrapper/wrap_cudadrv.cpp:504:48: note: expanded from macro 'DECLARE_EXC'
      Cuda##NAME = py::handle<>(PyErr_NewException("pycuda._driver." #NAME, BASE, NULL)); \
                                                   ^
    src/wrapper/wrap_cudadrv.cpp:511:5: warning: conversion from string literal to 'char *' is deprecated [-Wdeprecated-writable-strings]
        DECLARE_EXC(LaunchError, CudaError.get());
        ^
    src/wrapper/wrap_cudadrv.cpp:504:48: note: expanded from macro 'DECLARE_EXC'
      Cuda##NAME = py::handle<>(PyErr_NewException("pycuda._driver." #NAME, BASE, NULL)); \
                                                   ^
    src/wrapper/wrap_cudadrv.cpp:512:5: warning: conversion from string literal to 'char *' is deprecated [-Wdeprecated-writable-strings]
        DECLARE_EXC(RuntimeError, CudaError.get());
        ^
    src/wrapper/wrap_cudadrv.cpp:504:48: note: expanded from macro 'DECLARE_EXC'
      Cuda##NAME = py::handle<>(PyErr_NewException("pycuda._driver." #NAME, BASE, NULL)); \
                                                   ^
    src/wrapper/wrap_cudadrv.cpp:272:9: warning: unused function 'py_memcpy_peer_async' [-Wunused-function]
      void  py_memcpy_peer_async(CUdeviceptr dest, CUdeviceptr src,
            ^
    7 warnings generated.
    In file included from bpl-subset/bpl_subset/libs/thread/src/pthread/thread.cpp:27:
    bpl-subset/bpl_subset/libs/thread/src/pthread/timeconv.inl:48:13: warning: unused function 'to_time' [-Wunused-function]
    inline void to_time(int milliseconds, timespec& ts)
                ^
    bpl-subset/bpl_subset/libs/thread/src/pthread/timeconv.inl:86:13: warning: unused function 'to_duration' [-Wunused-function]
    inline void to_duration(boost::xtime xt, int& milliseconds)
                ^
    bpl-subset/bpl_subset/libs/thread/src/pthread/timeconv.inl:108:13: warning: unused function 'to_microduration' [-Wunused-function]
    inline void to_microduration(boost::xtime xt, int& microseconds)
                ^
    3 warnings generated.
    In file included from src/wrapper/mempool.cpp:2:
    In file included from src/wrapper/tools.hpp:10:
    In file included from src/wrapper/numpy_init.hpp:6:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/arrayobject.h:4:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/ndarrayobject.h:17:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/ndarraytypes.h:1760:
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: "Using deprecated NumPy API, disable it by "          "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-W#warnings]
#warning "Using deprecated NumPy API, disable it by " \
     ^
    1 warning generated.
    In file included from src/wrapper/_pvt_struct_v2.cpp:16:
    In file included from src/wrapper/numpy_init.hpp:6:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/arrayobject.h:4:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/ndarrayobject.h:17:
    In file included from /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/ndarraytypes.h:1760:
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: "Using deprecated NumPy API, disable it by "          "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-W#warnings]
#warning "Using deprecated NumPy API, disable it by " \
     ^
    src/wrapper/_pvt_struct_v2.cpp:131:30: warning: conversion from string literal to 'char *' is deprecated [-Wdeprecated-writable-strings]
    static char *integer_codes = "bBhHiIlLqQ";
                                 ^
    src/wrapper/_pvt_struct_v2.cpp:189:3: warning: conversion from string literal to 'char *' is deprecated [-Wdeprecated-writable-strings]
            {"format", (getter)s_get_format, (setter)NULL, "struct format string", NULL},
             ^
    src/wrapper/_pvt_struct_v2.cpp:189:49: warning: conversion from string literal to 'char *' is deprecated [-Wdeprecated-writable-strings]
            {"format", (getter)s_get_format, (setter)NULL, "struct format string", NULL},
                                                           ^
    src/wrapper/_pvt_struct_v2.cpp:190:3: warning: conversion from string literal to 'char *' is deprecated [-Wdeprecated-writable-strings]
            {"size", (getter)s_get_size, (setter)NULL, "struct size in bytes", NULL},
             ^
    src/wrapper/_pvt_struct_v2.cpp:190:45: warning: conversion from string literal to 'char *' is deprecated [-Wdeprecated-writable-strings]
            {"size", (getter)s_get_size, (setter)NULL, "struct size in bytes", NULL},
                                                       ^
    src/wrapper/_pvt_struct_v2.cpp:1023:27: warning: conversion from string literal to 'char *' is deprecated [-Wdeprecated-writable-strings]
            static char *kwlist[] = {"format", 0};
                                     ^
    src/wrapper/_pvt_struct_v2.cpp:1122:27: warning: conversion from string literal to 'char *' is deprecated [-Wdeprecated-writable-strings]
            static char *kwlist[] = {"buffer", "offset", 0};
                                     ^
    src/wrapper/_pvt_struct_v2.cpp:1122:37: warning: conversion from string literal to 'char *' is deprecated [-Wdeprecated-writable-strings]
            static char *kwlist[] = {"buffer", "offset", 0};
                                               ^
    src/wrapper/_pvt_struct_v2.cpp:1126:21: warning: conversion from string literal to 'char *' is deprecated [-Wdeprecated-writable-strings]
            static char *fmt = "z#|n:unpack_from";
                               ^
    src/wrapper/_pvt_struct_v2.cpp:1565:36: warning: conversion from string literal to 'char *' is deprecated [-Wdeprecated-writable-strings]
                    StructError = PyErr_NewException("pycuda._pvt_struct.error", NULL, NULL);
                                                     ^
    11 warnings generated.

    Installed /usr/local/env/chroma_env/src/chroma/pycuda-2013.1.1-py2.7-macosx-10.9-x86_64.egg
    Searching for spnav
    Reading https://pypi.python.org/simple/spnav/
    Best match: spnav 0.9
    Downloading https://pypi.python.org/packages/source/s/spnav/spnav-0.9.tar.gz#md5=94dbb6444d308d60eb56f88f727b2fe6
    Processing spnav-0.9.tar.gz
    Writing /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-28IfRD/spnav-0.9/setup.cfg
    Running spnav-0.9/setup.py -q bdist_egg --dist-dir /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-28IfRD/spnav-0.9/egg-dist-tmp-VTPNwI
    zip_safe flag not set; analyzing archive contents...

    Installed /usr/local/env/chroma_env/src/chroma/spnav-0.9-py2.7.egg
    Searching for pyzmq-static
    Reading https://pypi.python.org/simple/pyzmq-static/
    Best match: pyzmq-static 2.2
    Downloading https://pypi.python.org/packages/source/p/pyzmq-static/pyzmq-static-2.2.tar.gz#md5=42de12272357776b4ce38f3b9be8ca80
    Processing pyzmq-static-2.2.tar.gz
    Writing /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-TmpTC4/pyzmq-static-2.2/setup.cfg
    Running pyzmq-static-2.2/setup.py -q bdist_egg --dist-dir /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-TmpTC4/pyzmq-static-2.2/egg-dist-tmp-i2uP6R
    warning: install_lib: 'build/lib' does not exist -- no Python modules to install

    zip_safe flag not set; analyzing archive contents...

    Installed /usr/local/env/chroma_env/src/chroma/pyzmq_static-2.2-py2.7.egg
    Searching for uncertainties
    Reading https://pypi.python.org/simple/uncertainties/
    Best match: uncertainties 2.4.4
    Downloading https://pypi.python.org/packages/source/u/uncertainties/uncertainties-2.4.4.tar.gz#md5=77fc7ef882cb6e8488f092ea8abdf533
    Processing uncertainties-2.4.4.tar.gz
    Writing /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-7MSga8/uncertainties-2.4.4/setup.cfg
    Running uncertainties-2.4.4/setup.py -q bdist_egg --dist-dir /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-7MSga8/uncertainties-2.4.4/egg-dist-tmp-nnMQxs
    zip_safe flag not set; analyzing archive contents...
    uncertainties.lib1to2.test_1to2: module references __file__

    Installed /usr/local/env/chroma_env/src/chroma/uncertainties-2.4.4-py2.7.egg
    Searching for Jinja2>=2.3
    Reading https://pypi.python.org/simple/Jinja2/
    Best match: Jinja2 2.7.2
    Downloading https://pypi.python.org/packages/source/J/Jinja2/Jinja2-2.7.2.tar.gz#md5=df1581455564e97010e38bc792012aa5
    Processing Jinja2-2.7.2.tar.gz
    Writing /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-BJ0iXa/Jinja2-2.7.2/setup.cfg
    Running Jinja2-2.7.2/setup.py -q bdist_egg --dist-dir /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-BJ0iXa/Jinja2-2.7.2/egg-dist-tmp-wiCRed
    warning: no files found matching '*' under directory 'custom_fixers'
    warning: no previously-included files matching '*' found under directory 'docs/_build'
    warning: no previously-included files matching '*.pyc' found under directory 'jinja2'
    warning: no previously-included files matching '*.pyc' found under directory 'docs'
    warning: no previously-included files matching '*.pyo' found under directory 'jinja2'
    warning: no previously-included files matching '*.pyo' found under directory 'docs'

    Installed /usr/local/env/chroma_env/src/chroma/Jinja2-2.7.2-py2.7.egg
    Searching for Pygments>=1.2
    Reading https://pypi.python.org/simple/Pygments/
    Best match: Pygments 1.6
    Downloading https://pypi.python.org/packages/2.7/P/Pygments/Pygments-1.6-py2.7.egg#md5=1e1e52b1e434502682aab08938163034
    Processing Pygments-1.6-py2.7.egg
    creating /usr/local/env/chroma_env/src/chroma/Pygments-1.6-py2.7.egg
    Extracting Pygments-1.6-py2.7.egg to /usr/local/env/chroma_env/src/chroma

    Installed /usr/local/env/chroma_env/src/chroma/Pygments-1.6-py2.7.egg
    Searching for decorator>=3.2.0
    Reading https://pypi.python.org/simple/decorator/
    Best match: decorator 3.4.0
    Downloading https://pypi.python.org/packages/source/d/decorator/decorator-3.4.0.tar.gz#md5=1e8756f719d746e2fc0dd28b41251356
    Processing decorator-3.4.0.tar.gz
    Writing /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-ufp2dl/decorator-3.4.0/setup.cfg
    Running decorator-3.4.0/setup.py -q bdist_egg --dist-dir /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-ufp2dl/decorator-3.4.0/egg-dist-tmp-X5JCrT
    warning: no previously-included files found matching 'Makefile'

    Installed /usr/local/env/chroma_env/src/chroma/decorator-3.4.0-py2.7.egg
    Searching for pytools>=2011.2
    Reading https://pypi.python.org/simple/pytools/
    Best match: pytools 2013.5.7
    Downloading https://pypi.python.org/packages/source/p/pytools/pytools-2013.5.7.tar.gz#md5=8954a655749d646d456335b93aad3caf
    Processing pytools-2013.5.7.tar.gz
    Writing /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-nAx6JK/pytools-2013.5.7/setup.cfg
    Running pytools-2013.5.7/setup.py -q bdist_egg --dist-dir /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-nAx6JK/pytools-2013.5.7/egg-dist-tmp-ggsJaC
    zip_safe flag not set; analyzing archive contents...
    pytools.__init__: module MAY be using inspect.getouterframes
    pytools.debug: module MAY be using inspect.getouterframes
    pytools.diskdict: module references __file__

    Installed /usr/local/env/chroma_env/src/chroma/pytools-2013.5.7-py2.7.egg
    Searching for pyzmq
    Best match: pyzmq static-2.2
    Downloading https://pypi.python.org/packages/source/p/pyzmq-static/pyzmq-static-2.2.tar.gz#md5=42de12272357776b4ce38f3b9be8ca80
    Processing pyzmq-static-2.2.tar.gz
    Writing /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-8srkDM/pyzmq-static-2.2/setup.cfg
    Running pyzmq-static-2.2/setup.py -q bdist_egg --dist-dir /var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/easy_install-8srkDM/pyzmq-static-2.2/egg-dist-tmp-UQp4qP
    warning: install_lib: 'build/lib' does not exist -- no Python modules to install

    zip_safe flag not set; analyzing archive contents...

    Installed /usr/local/env/chroma_env/src/chroma/pyzmq_static-2.2-py2.7.egg
    Traceback (most recent call last):
      File "setup.py", line 88, in <module>
        test_suite = 'nose.collector',
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/distutils/core.py", line 152, in setup
        dist.run_commands()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/distutils/dist.py", line 953, in run_commands
        self.run_command(cmd)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/distutils/dist.py", line 972, in run_command
        cmd_obj.run()
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/command/test.py", line 128, in run
        self.distribution.fetch_build_eggs(self.distribution.install_requires)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/setuptools/dist.py", line 263, in fetch_build_eggs
        parse_requirements(requires), installer=self.fetch_build_egg
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pkg_resources.py", line 572, in resolve
        raise DistributionNotFound(req)
    pkg_resources.DistributionNotFound: pyzmq




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




