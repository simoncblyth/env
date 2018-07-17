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


M$ bash
--------

* http://techcrunch.com/2016/03/30/be-very-afraid-hell-has-frozen-over-bash-is-coming-to-windows-10/

Investigating, this a bit : it isnt all that useful as not integrated with the rest of the system.   


Virtual Windows on Mac ?
--------------------------

* http://www.tekrevue.com/2015-vm-benchmarks-parallels-11-vs-fusion-8/16/

  Fusion is ahead, but still a long way behind direct usage with Bootcamp

Mercurial On Windows
---------------------

* https://www.mercurial-scm.org/wiki/WindowsInstall

  Too painful to go thru this kinda thing for every tool, ... need a distro  

Windows Cmd Line
------------------

To start: windows-R, type cmd return 

Windows Batch .cmd Scripting
------------------------------

http://steve-jansen.github.io/guides/windows-batch-scripting/part-1-getting-started.html  

Very limited, have to fake functions with goto. 


EOU
}
win-dir(){ echo $(local-base)/env/windows/windows-win ; }
win-cd(){  cd $(win-dir); }
win-mate(){ mate $(win-dir) ; }
win-get(){
   local dir=$(dirname $(win-dir)) &&  mkdir -p $dir && cd $dir

}
