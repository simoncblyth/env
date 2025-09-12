# === func-gen- : tools/nanobind fgp tools/nanobind.bash fgn nanobind fgh tools src base/func.bash
nanobind-source(){   echo ${BASH_SOURCE} ; }
nanobind-edir(){ echo $(dirname $(nanobind-source)) ; }
nanobind-ecd(){  cd $(nanobind-edir); }
nanobind-dir(){  echo $LOCAL_BASE/env/tools/nanobind ; }
nanobind-cd(){   cd $(nanobind-dir); }
nanobind-vi(){   vi $(nanobind-source) ; }
nanobind-env(){  elocal- ; }
nanobind-usage(){ cat << EOU

nanobind
==========

refs
-----

* https://github.com/wjakob/nanobind
* https://nanobind.readthedocs.io/en/latest/
* https://nanobind.readthedocs.io/_/downloads/en/latest/pdf/

* https://nanobind.readthedocs.io/en/latest/installing.html
* https://nanobind.readthedocs.io/en/latest/exchanging.html
* https://nanobind.readthedocs.io/en/latest/ownership.html

* https://nanobind.readthedocs.io/en/latest/ndarray.html


binding discussion
-------------------

* https://discuss.python.org/t/ideas-for-forward-compatible-and-fast-extension-libraries-in-python-3-12/15993


usage
-------

* https://github.com/search?q=nanobind&type=repositories


* https://github.com/leapmotion/pyopticam/blob/main/src/pyopticam_ext.cpp

  * mostly just listing of methods with lambdas


* https://github.com/LabSound/LabSound/blob/da521feba56d1d9dd4ab66bdb4ebcebaee83e40d/labsoundpy/CMakeLists.txt#L37



nanobind_add_module
--------------------

* https://nanobind.readthedocs.io/en/latest/api_cmake.html


nanobind examples
------------------

* most proj either too big or too small to act as good examples
* majority of bindings seem like a whole other development rather than
  just being a connection between C++ and python



github usage of "nanobind_add_module"
---------------------------------------

* https://github.com/search?q=nanobind_add_module&type=code

mitsuba3
----------

* https://github.com/mitsuba-renderer/mitsuba3/blob/6a89b9df132d6ff869e6e9384facbc229ad33917/src/python/CMakeLists.txt#L33


mlx
-----

* https://ml-explore.github.io/mlx/build/html/index.html
* https://ml-explore.github.io/mlx/build/html/dev/extensions.html

* https://ml-explore.github.io/mlx/build/html/cpp/ops.html#cpp-ops
* https://ml-explore.github.io/mlx/build/html/python/ops.html#ops



* https://github.com/ml-explore/mlx/blob/main/CMakeLists.txt

::

    if(MLX_BUILD_PYTHON_BINDINGS)
      message(STATUS "Building Python bindings.")
      find_package(
        Python 3.8
        COMPONENTS Interpreter Development.Module
        REQUIRED)
      execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE nanobind_ROOT)
      find_package(nanobind CONFIG REQUIRED)
      add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/python/src)
    endif()


* https://github.com/ml-explore/mlx/blob/main/python/src/CMakeLists.txt

::

    nanobind_add_module(
      core
      NB_STATIC
      STABLE_ABI
      LTO
      NOMINSIZE
      NB_DOMAIN
      mlx
      ${CMAKE_CURRENT_SOURCE_DIR}/mlx.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/array.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/convert.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/distributed.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/export.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/fast.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/load.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/metal.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/cuda.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/memory.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/mlx_func.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/ops.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/stream.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/transforms.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/random.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/linalg.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/constants.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/trees.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp)

    ...
    target_link_libraries(core PRIVATE mlx)

    if(BUILD_SHARED_LIBS)
      if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        target_link_options(core PRIVATE -Wl,-rpath,@loader_path/lib)
      else()
        target_link_options(core PRIVATE -Wl,-rpath,\$ORIGIN/lib)
      endif()
    endif()


::

    find . -type f -exec grep -l nanobind {} \;


    ./.circleci/config.yml
    ./CMakeLists.txt
    ./docs/src/conf.py
    ./docs/src/dev/extensions.rst
    ./examples/extensions/CMakeLists.txt
    ./examples/extensions/bindings.cpp
    ./examples/extensions/pyproject.toml
    ./examples/extensions/requirements.txt
    ./pyproject.toml
    ./python/src/CMakeLists.txt
    ./python/src/array.cpp
    ./python/src/buffer.h
    ./python/src/constants.cpp
    ./python/src/convert.cpp
    ./python/src/convert.h
    ./python/src/cuda.cpp
    ./python/src/device.cpp
    ./python/src/distributed.cpp
    ./python/src/export.cpp
    ./python/src/fast.cpp
    ./python/src/fft.cpp
    ./python/src/indexing.h
    ./python/src/linalg.cpp
    ./python/src/load.cpp
    ./python/src/load.h
    ./python/src/memory.cpp
    ./python/src/metal.cpp
    ./python/src/mlx.cpp
    ./python/src/mlx_func.cpp
    ./python/src/mlx_func.h
    ./python/src/ops.cpp
    ./python/src/random.cpp
    ./python/src/small_vector.h
    ./python/src/stream.cpp
    ./python/src/transforms.cpp
    ./python/src/trees.h
    ./python/src/utils.h
    ./setup.py





pytensor (SKBUILD)
---------------------

* https://github.com/patrickroberts/pytensor/tree/main

::

   scikit-build-core >=0.10.5




NVlabs/flip
-----------


* https://github.com/NVlabs/flip
* https://github.com/NVlabs/flip/blob/main/CMakeLists.txt

::

    nanobind_add_module(nbflip STABLE_ABI src/nanobindFLIP.cpp)
    install(TARGETS nbflip LIBRARY DESTINATION flip_evaluator)


* https://github.com/NVlabs/flip/blob/main/src/nanobindFLIP.cpp



evanwporter/Cloth (pandas-lite esp for timeseries data)
-----------------------------------------------------------

* https://github.com/evanwporter/Cloth/blob/main/src/cloth.cpp




mcneel/rhino3dm
----------------


* https://pypi.org/project/rhino3dm/
* https://stevebaer.wordpress.com/2018/10/15/rhino3dm-geometry-toolkits-for-net-python-and-javascript/

* https://github.com/mcneel/rhino3dm
* https://github.com/mcneel/rhino3dm/blob/dev/src/CMakeLists.txt#L112

::


    file(GLOB bindings_SRC "bindings/*.h" "bindings/*.cpp")
    file(GLOB zlib_SRC "lib/opennurbs/zlib/*.h" "lib/opennurbs/zlib/*.c")

    # temporarily rename the 3 cpp files that we don't want to compile on OSX
    file(RENAME "lib/opennurbs/opennurbs_gl.cpp" "lib/opennurbs/opennurbs_gl.skip")
    file(RENAME "lib/opennurbs/opennurbs_unicode_cp932.cpp" "lib/opennurbs/opennurbs_unicode_cp932.skip")
    file(RENAME "lib/opennurbs/opennurbs_unicode_cp949.cpp" "lib/opennurbs/opennurbs_unicode_cp949.skip")
    file(GLOB opennurbs_SRC "lib/opennurbs/*.h" "lib/opennurbs/*.cpp")
    file(RENAME "lib/opennurbs/opennurbs_gl.skip" "lib/opennurbs/opennurbs_gl.cpp")
    file(RENAME "lib/opennurbs/opennurbs_unicode_cp932.skip" "lib/opennurbs/opennurbs_unicode_cp932.cpp")
    file(RENAME "lib/opennurbs/opennurbs_unicode_cp949.skip" "lib/opennurbs/opennurbs_unicode_cp949.cpp")


    if (${RHINO3DM_PY})
      if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_definitions(-DON_RUNTIME_LINUX)
        add_definitions(-DON_CLANG_CONSTRUCTOR_BUG)
        if(PYTHON_BINDING_LIB STREQUAL "NANOBIND")
          #need to build nanobind to link it?
          nanobind_build_library(nanobind)
          nanobind_add_module(_rhino3dm NB_STATIC ${bindings_SRC} ${opennurbs_SRC} ${zlib_SRC} ${uuid_SRC})
          target_link_libraries(_rhino3dm PRIVATE nanobind-static)
        else()
          pybind11_add_module(_rhino3dm ${bindings_SRC} ${opennurbs_SRC} ${zlib_SRC} ${uuid_SRC})
          target_link_libraries(_rhino3dm PRIVATE pybind11::module)
        endif()


* https://github.com/mcneel/rhino3dm/blob/dev/src/bindings/bindings.h

::

    #include "bnd_color.h"
    #include "bnd_file_utilities.h"
    #include "bnd_uuid.h"
    #include "bnd_defines.h"
    #include "bnd_intersect.h"
    #include "bnd_boundingbox.h"
    #include "bnd_box.h"
    #include "bnd_point.h"
    #include "bnd_object.h"
    #include "bnd_geometry.h"
    #include "bnd_curve.h"
    #include "bnd_linecurve.h"
    ...




::

    A[blyth@localhost rhino3dm]$ find . -type f -exec grep -l nanobind {} \;
    ...
    ./.gitmodules
    ./CHANGELOG.md
    ./src/CMakeLists.txt
    ./src/bindings/bindings.h
    ./src/bindings/bnd_object.cpp
    ./src/lib/README.md
    ./src/rhinocore_bindings/CMakeLists.txt
    A[blyth@localhost rhino3dm]$


src/rhinocore_bindings/CMakeLists.txt::

     09 file(GLOB bindings_SRC "../bindings/*.h" "../bindings/*.cpp")
     10
     11 set(RHINO3DM_PY "YES")
     12 set(NANOBIND_BUILD "YES")
     13
     14 if (${NANOBIND_BUILD})
     15   add_compile_definitions( NANOBIND )
     16   if (CMAKE_VERSION VERSION_LESS 3.18)
     17     set(DEV_MODULE Development)
     18   else()
     19     set(DEV_MODULE Development.Module)
     20   endif()
     21   find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
     22
     23   if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
     24     set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
     25     set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
     26   endif()
     27   add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../lib/nanobind nanobind_dir)
     28   nanobind_add_module(rhinocore_py_bindings ${bindings_SRC} stdafx.cpp)
     29 else()
     30   add_subdirectory(../lib/pybind11 pybind11_dir)
     31   pybind11_add_module(rhinocore_py_bindings ${bindings_SRC} stdafx.cpp)
     32 endif()








GPUOpen-LibrariesAndSDKs/RadeonProRenderSDK
-----------------------------------------------

* https://github.com/GPUOpen-LibrariesAndSDKs/RadeonProRenderSDK
* https://github.com/GPUOpen-LibrariesAndSDKs/RadeonProRenderSDK/blob/master/python/CMakeLists.txt

::

    nanobind_add_module(rpr bind_rpr.cpp bind_common.cpp bind_common.h)
    nanobind_add_module(rprs bind_rprs.cpp bind_common.cpp bind_common.h)
    nanobind_add_module(rprgltf bind_rprgltf.cpp bind_common.cpp bind_common.h)

    if (LINUX)
    target_link_libraries(rpr PUBLIC RadeonProRender64)
    target_link_libraries(rprs PUBLIC RprLoadStore64)
    target_link_libraries(rprgltf PUBLIC ProRenderGLTF)
    endif (LINUX)

* https://github.com/GPUOpen-LibrariesAndSDKs/RadeonProRenderSDK/tree/master/python

* https://github.com/GPUOpen-LibrariesAndSDKs/RadeonProRenderSDK/blob/master/python/bind_rpr.cpp


::

	nb::class_<rpr_image_desc>(m, "ImageDesc")
.def(nb::init<>())
		.def_rw("width", &rpr_image_desc::image_width)
		.def_rw("height", &rpr_image_desc::image_height)
		.def_rw("depth", &rpr_image_desc::image_depth)
		.def_rw("row_pitch", &rpr_image_desc::image_row_pitch)
		.def_rw("slice_pitch", &rpr_image_desc::image_slice_pitch)
	;




ilia-kats/numpy-onlinestats (SKBUILD) : SIMPLE AND ON PYPI
-----------------------------------------------------------

* https://github.com/ilia-kats/numpy-onlinestats/blob/master/CMakeLists.txt=
* https://github.com/ilia-kats/numpy-onlinestats/tree/master/src

::

    RunningStats.{h,cpp}  ## no binding stuff at all


* https://numpy-onlinestats.readthedocs.io/en/latest/




* https://github.com/mcneel/rhino3dm



cupy-nanobind-example
-----------------------

* https://github.com/lgarrison/cupy-nanobind-example/blob/main/src/example/cuda.cu


conda nanobind scikit-build-core
----------------------------------

* https://github.com/Krande/nanobind-minimal



Separate nanobind-ing of NumPy and OpenCV
---------------------------------------------------

* https://github.com/pthom/cvnp_nano



llama.cpp nanobinding
------------------------

*  https://github.com/shakfu/llamalib


uv workspaces ?
-----------------

* https://docs.astral.sh/uv/concepts/projects/workspaces/

Inspired by the Cargo concept of the same name, a workspace is "a collection of
one or more packages, called workspace members, that are managed together."

Workspaces organize large codebases by splitting them into multiple packages
with common dependencies. Think: a FastAPI-based web application, alongside a
series of libraries that are versioned and maintained as separate Python
packages, all in the same Git repository.

* https://github.com/jurihock/nanobind_uv_workspace_example/blob/main/pyproject.toml



avoids the nasty def chain
----------------------------

* https://opensource.adobe.com/lagrange-docs/cpp/bind__surface__mesh_8h_source.html


nanobind ndarray examples
--------------------------

* https://stackoverflow.com/questions/78777991/return-ndarray-in-nanobind-with-owned-memory
* https://nanobind.readthedocs.io/en/latest/ndarray.html#data-ownership



structure of large scale nanobinding
----------------------------------------

* https://github.com/StudioWEngineers/py3dm
* https://github.com/StudioWEngineers/py3dm/blob/main/src/bindings/bindings.cpp


nanobind_example of scikit-build-core + CMake + nanobind packaging with GHA GitHubActions
-------------------------------------------------------------------------------------------

* https://github.com/wjakob/nanobind_example
* https://github.com/wjakob/nanobind_example/blob/master/CMakeLists.txt
* https://github.com/wjakob/nanobind_example/blob/master/pyproject.toml

pyproject.toml::

    [build-system]
    requires = ["scikit-build-core >=0.10", "nanobind >=1.3.2"]
    build-backend = "scikit_build_core.build"


* https://cibuildwheel.pypa.io/en/stable/


packaging
------------

* https://nanobind.readthedocs.io/en/latest/packaging.html


A Python wheel is a self-contained binary file that bundles Python code and
extension libraries along with metadata such as versioned package dependencies.
Wheels are easy to download and install, and they are the recommended mechanism
for distributing extensions created using nanobind.

scikit-build-core integrated CMake/python wheel building/packaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://proceedings.scipy.org/articles/FMKR8387
* https://scikit-build-core.readthedocs.io/en/latest/




binding
---------


* https://nanobind.readthedocs.io/en/latest/basics.html#basics

+-------------------------+----------------------------------------------+
| Methods & constructors  | .def()                                       |
+-------------------------+----------------------------------------------+
| Fields                  | .def_ro(), .def_rw()                         |
+-------------------------+----------------------------------------------+
| Properties              | .def_prop_ro(), .def_prop_rw()               |
+-------------------------+----------------------------------------------+
| Static methods          | .def_static()                                |
+-------------------------+----------------------------------------------+
| Static fields           | .def_ro_static(), .def_rw_static()           |
+-------------------------+----------------------------------------------+
| Static properties       | .def_prop_ro_static(), .def_prop_rw_static() |
+-------------------------+----------------------------------------------+



Exploration : /usr/local/env/nanobind_check
---------------------------------------------

::

    mkdir /usr/local/env/nanobind_check
    cd /usr/local/env/nanobind_check
    uv venv
    uv pip install nanobind

::

    zeta:nanobind_check blyth$ uv pip install nanobind
    Resolved 1 package in 1.17s
    Prepared 1 package in 673ms
    Installed 1 package in 3ms
     + nanobind==2.9.2
    zeta:nanobind_check blyth$

Everything in the .venv::

    zeta:nanobind_check blyth$ l
    0 drwxr-xr-x  8 blyth  staff  256 Sep 10 16:34 .venv

    zeta:nanobind_check blyth$ uv pip list
    Package  Version
    -------- -------
    nanobind 2.9.2




ndarray
----------

* https://nanobind.readthedocs.io/en/latest/ndarray.html

You can think of the nb::ndarray class as a reference-counted pointer
resembling std::shared_ptr<T> that can be freely moved or copied. This means
that there isn’t a big difference between a function taking ndarray by value
versus taking a constant reference const ndarray & (i.e., the former does not
create an additional copy of the underlying data).

Copies of the nb::ndarray wrapper will point to the same underlying buffer and
increase the reference count until they go out of scope. You may call freely
call nb::ndarray<...> methods from multithreaded code even when the GIL is not
held, for example to examine the layout of an array and access the underlying
storage.

There are two exceptions to this: creating a new nd-array object from C++
(discussed later) and casting it to Python via the ndarray::cast() function
both involve Python API calls that require that the GIL is held.




nanobind vs pybind11 vs CPython
---------------------------------

* https://ashvardanian.com/posts/pybind11-cpython-tutorial/

Speed is not an issue for me as only need to bind at a very high level
of doing an array transform eg from gensteps to hits.


* https://news.ycombinator.com/item?id=44581631


alt : cppyy
-------------

* https://cppyy.readthedocs.io/en/latest/

The main downside of cppyy is that it depends on Cling/Clang/LLVM that must be
deployed on the user side and then run there. There isnt a way of
pre-generating bindings and then shipping just the output of this process.









EOU
}
nanobind-get(){
   local dir=$(dirname $(nanobind-dir)) &&  mkdir -p $dir && cd $dir

}
