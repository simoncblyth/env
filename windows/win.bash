# === func-gen- : windows/win fgp windows/win.bash fgn win fgh windows
win-src(){      echo windows/win.bash ; }
win-source(){   echo ${BASH_SOURCE:-$(env-home)/$(win-src)} ; }
win-vi(){       vi $(win-source) ; }
win-env(){      elocal- ; }
win-usage(){ cat << EOU

Windows
=========

command prompt
----------------

* http://www.digitalcitizen.life/7-ways-launch-command-prompt-windows-7-windows-8

compiling from command prompt
------------------------------

* https://msdn.microsoft.com/en-us/library/bb384838.aspx

cmake
-------

* https://cmake.org/runningcmake/

Example of windows command line building with cmake

* https://trac.osgeo.org/geos/wiki/BuildingOnWindowsWithCMake

* http://stackoverflow.com/questions/1459482/how-to-use-cmake-for-non-interactive-build-on-windows


EOU
}
win-dir(){ echo $(local-base)/env/windows/windows-win ; }
win-cd(){  cd $(win-dir); }
win-mate(){ mate $(win-dir) ; }
win-get(){
   local dir=$(dirname $(win-dir)) &&  mkdir -p $dir && cd $dir

}
