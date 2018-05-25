# === func-gen- : tools/bcm fgp tools/bcm.bash fgn bcm fgh tools
bcm-src(){      echo tools/bcm.bash ; }
bcm-source(){   echo ${BASH_SOURCE:-$(env-home)/$(bcm-src)} ; }
bcm-vi(){       vi $(bcm-source) ; }
bcm-env(){      elocal- ; }
bcm-usage(){ cat << EOU

BCM : Boost CMake Modules
===========================

This provides cmake modules that can be re-used by boost and other
dependencies. It provides modules to reduce the boilerplate for installing,
versioning, setting up package config, and creating tests.

* https://github.com/boost-cmake/bcm
* https://readthedocs.org/projects/bcm/
* http://bcm.readthedocs.io/en/latest/

Very clear explanation describing a standalone CMake setup for building boost_filesystem

* http://bcm.readthedocs.io/en/latest/src/Building.html#building-standalone-with-cmake

bcm_auto_export
----------------

* https://github.com/boost-cmake/bcm/blob/master/share/bcm/cmake/BCMExport.cmake




EOU
}
bcm-dir(){ echo $(local-base)/env/tools/tools-bcm ; }
bcm-cd(){  cd $(bcm-dir); }
bcm-mate(){ mate $(bcm-dir) ; }
bcm-get(){
   local dir=$(dirname $(bcm-dir)) &&  mkdir -p $dir && cd $dir

}
