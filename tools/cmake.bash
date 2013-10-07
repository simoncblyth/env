# === func-gen- : tools/cmake fgp tools/cmake.bash fgn cmake fgh tools
cmake-src(){      echo tools/cmake.bash ; }
cmake-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cmake-src)} ; }
cmake-vi(){       vi $(cmake-source) ; }
cmake-env(){      elocal- ; }
cmake-usage(){ cat << EOU

CMAKE
======

* http://www.cmake.org/

EXTERNAL LIBS
--------------

* http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries
* http://www.cmake.org/cmake/help/v2.8.8/cmake.html#command:find_package

::

    [blyth@belle7 gdml2wrl]$ cmake --help-module-list | grep Find
    CMakeFindFrameworks
    FindASPELL
    FindAVIFile
    FindBLAS
    FindBZip2
    FindBoost
    FindCABLE
    FindCURL
    FindCVS
    FindCoin3D


Geant4.cmake
~~~~~~~~~~~~~

* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/InstallationGuide/html/ch03s02.html







EOU
}
cmake-dir(){ echo $(local-base)/env/tools/tools-cmake ; }
cmake-cd(){  cd $(cmake-dir); }
cmake-mate(){ mate $(cmake-dir) ; }
cmake-get(){
   local dir=$(dirname $(cmake-dir)) &&  mkdir -p $dir && cd $dir

}
