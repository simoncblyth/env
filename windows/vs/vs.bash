# === func-gen- : windows/vs/vs fgp windows/vs/vs.bash fgn vs fgh windows/vs
vs-src(){      echo windows/vs/vs.bash ; }
vs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vs-src)} ; }
vs-vi(){       vi $(vs-source) ; }
vs-env(){      elocal- ; }
vs-usage(){ cat << \EOU

Microsoft Visual Studio 2015 Community Edition
================================================

VS Versions
-------------

xerces-c 3.1.3 comes with various .sln files 
corresponding to different visual C++ compilers 
that come with various versions of visual studio

Unfortunately the lastest is for VC12, not VC14::

   VC14  Visual Studio 2015
   VC12  Visual Studio 2013
   VC11  Visual Studio 2012
   VC10  Visual Studio 2010


* https://en.wikipedia.org/wiki/Microsoft_Visual_Studio

::

                               vers     cl.exe   .NET supported  Release date
                               VC        vers     vers           
    Visual Studio 2012  Dev11   11.0    17.00   2.0 – 4.5.2      September 12, 2012
    Visual Studio 2013  Dev12   12.0    18.00   2.0 – 4.5.2      October 17, 2013
    Visual Studio 2015  Dev14   14.0    19.00   2.0 – 4.6        July 20, 2015
    Visual Studio "15"  Dev15   15.0    19.00   2.0 – 5.0        March 30, 2016



VS Redist
----------

The Visual C++ Redistributable Packages install run-time components that are
required to run C++ applications built using Visual Studio 2015.

* https://www.microsoft.com/en-us/download/details.aspx?id=48145



Windows Building On Command Line
-----------------------------------

* https://msdn.microsoft.com/en-us/library/f35ctcxw.aspx?f=255&MSPPError=-2147217396

Windows Makefiles/CMake
--------------------------

Modifying a simple Linux Makefile to work on windows

* https://cognitivewaves.wordpress.com/makefiles-windows/

VS generation example

* https://cognitivewaves.wordpress.com/cmake-and-visual-studio/



Windows Development Environment
---------------------------------

* :google:`Windows Development Environment`

* https://github.com/felixrieseberg/windows-development-environment

Install Chocolatey
~~~~~~~~~~~~~~~~~~~

* see chocolatey-

Find Command Prompt (Windows 7 SP1)
-------------------------------------

* Start Menu, triangle expand "All Programs"
* right click "Visual Studio 2015" and Expand, and again on "Visual Studio Tools"

::

    Developer Command Prompt for VS2015
    MSBuild Command Prompt 


Standard Windows cmd.exe
--------------------------

* windows-key R (for Run menu)
* type "cmd" and ok

From cmd.exe shell entering @powershell gets you there.


Run cmd.exe as administrator
------------------------------

* Start > All Programs > Accessories
* Right click "Command Prompt" and choose "Run as administrator" answer Y to user access control question, 
* this gets to a prompt that looks exactly like non-admin one



cl compiler
-------------

* all three of above prompts didnt recognize cl 

git 
----

* seems git can be used from IDE, but not exposed from shells

powershell git : posh-git
---------------------------

* https://github.com/dahlbyk/posh-git




EOU
}
vs-dir(){ echo $(local-base)/env/windows/vs/windows/vs-vs ; }
vs-cd(){  cd $(vs-dir); }
vs-mate(){ mate $(vs-dir) ; }
vs-get(){
   local dir=$(dirname $(vs-dir)) &&  mkdir -p $dir && cd $dir

}
