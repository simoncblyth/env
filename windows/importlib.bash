# === func-gen- : windows/importlib fgp windows/importlib.bash fgn importlib fgh windows
importlib-src(){      echo windows/importlib.bash ; }
importlib-source(){   echo ${BASH_SOURCE:-$(env-home)/$(importlib-src)} ; }
importlib-vi(){       vi $(importlib-source) ; }
importlib-env(){      elocal- ; }
importlib-usage(){ cat << EOU

Windows Import Libs and DLLS
==============================

Intro
-------

Excellent description of windows library peculiarities.

* http://gernotklingler.com/blog/creating-using-shared-libraries-different-compilers-different-operating-systems/

CMake
-------

* https://cmake.org/cmake/help/v3.3/module/GenerateExportHeader.html
* https://cmake.org/Wiki/BuildingWinDLL
* https://blog.kitware.com/create-dlls-on-windows-without-declspec-using-new-cmake-export-all-feature/

  CMake 3.4 will have a new feature to simplify porting C and C++ software using shared libraries from Linux/UNIX to Windows

* http://stackoverflow.com/questions/33062728/cmake-link-shared-library-on-windows
* http://stackoverflow.com/questions/7614286/how-do-i-get-cmake-to-create-a-dll-and-its-matching-lib-file

MS
----

* https://msdn.microsoft.com/en-us/library/ms235636.aspx

  Walkthrough: Creating and Using a Dynamic Link Library (C++)



EOU
}


importlib-url(){ echo https://github.com/gklingler/sharedLibsDemo ; }

importlib-nam(){ echo $(basename $(importlib-url)) ; }
importlib-dir(){ echo $(local-base)/env/windows/$(importlib-nam) ; }
importlib-cd(){  cd $(importlib-dir); }
importlib-get(){
   local dir=$(dirname $(importlib-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(importlib-url)
   local nam=$(basename $url)
   [ ! -d "$nam" ] && git clone $url
}



