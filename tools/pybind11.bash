# === func-gen- : tools/pybind11 fgp tools/pybind11.bash fgn pybind11 fgh tools src base/func.bash
pybind11-source(){   echo ${BASH_SOURCE} ; }
pybind11-edir(){ echo $(dirname $(pybind11-source)) ; }
pybind11-ecd(){  cd $(pybind11-edir); }
pybind11-dir(){  echo $LOCAL_BASE/env/tools/pybind11 ; }
pybind11-cd(){   cd $(pybind11-dir); }
pybind11-vi(){   vi $(pybind11-source) ; }
pybind11-env(){  elocal- ; }
pybind11-usage(){ cat << EOU



pybind/cmake_example
-----------------------

* https://github.com/pybind/cmake_example
* https://github.com/pybind/cmake_example/blob/master/CMakeLists.txt

::

    git clone --recursive https://github.com/pybind/cmake_example.git


    A[blyth@localhost cmake_example]$ CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" uv pip install .
    Resolved 1 package in 500ms
          Built cmake-example @ file:///usr/local/ai/cmake_example
    Prepared 1 package in 3.32s
    Installed 1 package in 1ms
     + cmake-example==0.0.1 (from file:///usr/local/ai/cmake_example)
    A[blyth@localhost cmake_example]$ 



pybind11 tute
-----------------

* https://levelup.gitconnected.com/pybind11-a-beginners-guide-to-setting-up-c-and-python-projects-1de2f04465e9


* https://pypi.org/project/example-python-extension-cpp/


* https://stackoverflow.com/questions/71127446/is-passing-numpy-arrays-to-c-via-pybind-supposed-to-be-this-slow


pybind11 passing numpy array to C++ and returning one
-------------------------------------------------------

* https://zingale.github.io/phy504/cxx-python-sum.html
* https://zingale.github.io/phy504/cxx-python-array.html


pybind11 numpy
-----------------

* https://scicoding.com/pybind11-numpy-compatible-arrays/
* https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html


* https://github.com/pybind/pybind11_json


nanobind
-----------

* https://github.com/wjakob/nanobind
* https://nanobind.readthedocs.io/en/latest/
* https://nanobind.readthedocs.io/en/latest/why.html
* https://nanobind.readthedocs.io/en/latest/porting.html#removed

* https://nanobind.readthedocs.io/en/latest/ndarray.html
* https://github.com/dmlc/dlpack




EOU
}
pybind11-get(){
   local dir=$(dirname $(pybind11-dir)) &&  mkdir -p $dir && cd $dir

}
