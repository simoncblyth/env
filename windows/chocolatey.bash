# === func-gen- : windows/chocolatey fgp windows/chocolatey.bash fgn chocolatey fgh windows
chocolatey-src(){      echo windows/chocolatey.bash ; }
chocolatey-source(){   echo ${BASH_SOURCE:-$(env-home)/$(chocolatey-src)} ; }
chocolatey-vi(){       vi $(chocolatey-source) ; }
chocolatey-env(){      elocal- ; }
chocolatey-usage(){ cat << EOU

Windows Package Manager
=========================

* chocolatey.org

* https://github.com/felixrieseberg/windows-development-environment

::


    @powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((new-object net.webclient).DownloadString('https://chocolatey.org/install.ps1'))" && SET PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin


Helpful commands::

   choco /?
   choco list --lo   # local only packages


Need to run as Administrator, then can install pkgs with::

   choco search hg
   choco install hg   # can choose to print the script to check urls before running



Check web for more info on packages eg:

* http://chocolatey.org/packages?q=hg
* http://chocolatey.org/packages?q=vim
* https://chocolatey.org/packages?q=gow


Git And Unix Tools
--------------------

::

   choco install git.install --params="'/GitAndUnixToolsOnPath'"


Usage Experience
---------------------

* For updates to PATH to be seen, need to start a new cmd.exe or powershell.

* Where choco installs is not controlled, and sometimes PATH is not updated.
  Places to look::

        C:\Program Files
        C:\Program Files (x86)
        C:\ProgramData\chocolatey\lib\foo
        C:\ProgramData\chocolatey\bin

* installing cmake failed to provide anything, but cmake.portable worked
  and cmake 3.5.2 is now available


EOU
}
chocolatey-dir(){ echo $(local-base)/env/windows/windows-chocolatey ; }
chocolatey-cd(){  cd $(chocolatey-dir); }
chocolatey-mate(){ mate $(chocolatey-dir) ; }
chocolatey-get(){
   local dir=$(dirname $(chocolatey-dir)) &&  mkdir -p $dir && cd $dir

}
