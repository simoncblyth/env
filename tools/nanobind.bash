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
