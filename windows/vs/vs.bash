# === func-gen- : windows/vs/vs fgp windows/vs/vs.bash fgn vs fgh windows/vs
vs-src(){      echo windows/vs/vs.bash ; }
vs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vs-src)} ; }
vs-vi(){       vi $(vs-source) ; }
vs-env(){      elocal- ; }
vs-usage(){ cat << \EOU

Microsoft Visual Studio 2015 Community Edition
================================================

Installation
--------------

C++ tools like the compiler cl.exe and vcvarsall.bat 
are not included as standard, need to do a custom install OR do 
an update as noted in the below.

* https://blogs.msdn.microsoft.com/vcblog/2015/07/24/setup-changes-in-visual-studio-2015-affecting-c-developers/




Opening VS On a sln from Powershell
---------------------------------------

::

    PS> imp vs    # this is done by $profile usually 
    PS> vs-export # environment setup

    PS> C:\usr\local\env\windows\importclient\build> devenv .\DemoClient.sln


To ease transition from gitbash to powershell echo sln winpath 

    $ importclient-slnwin
    C:\usr\local\env\windows\importclient\build\DemoClient.sln

Then can::

    devenv C:\usr\local\env\windows\importclient\build\DemoClient.sln 




Using VS GUI with A CMake Generated Solution
-----------------------------------------------

* https://cognitivewaves.wordpress.com/cmake-and-visual-studio/

* in right pane "Solution Explorer" rightclick 
  the relevant target and  "Set As Startup project" 


Setting PATH for running debugger
----------------------------------------

Project > Properties > [Debugging]

For "Environment" field set 

* PATH=C:\usr\local\env\windows\sharedLibsDemo\build-windows-msvc\Release;%PATH%;$(LocalDebuggerEnvironment)
* NB delimiters are semi-colons


Setting Breakpoint
-------------------

Navigate to some source, right click on a line and choose "Set Breakpoint"


Start/Continue Debugger : F5
---------------------------------

A console window should show up with the output.



Windows LDD equivalent
------------------------

* http://stackoverflow.com/questions/7378959/how-to-check-for-dll-dependency 
* http://dependencywalker.com/


::

    PS > vs-export

    PS C:\usr\local\opticks\lib> dumpbin /dependents .\BoostRapClient.exe
    Microsoft (R) COFF/PE Dumper Version 14.00.23918.0
    Copyright (C) Microsoft Corporation.  All rights reserved.


    Dump of file .\BoostRapClient.exe

    File Type: EXECUTABLE IMAGE

      Image has the following dependencies:

        BoostRap.dll
        MSVCP140D.dll
        VCRUNTIME140D.dll
        ucrtbased.dll
        KERNEL32.dll






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


developer command prompt
----------------------------

From Start search for "Developer Command Prompt for VS2015"

Unfortunately thats not powershell. but can setup env::


    PS C:\Check2> cd $env:VS140COMNTOOLS/../../VC/bin
    PS C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin> ls


        Directory: C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin


    Mode                LastWriteTime     Length Name
    ----                -------------     ------ ----
    d----         6/12/2016   1:02 AM            1033
    d----         6/12/2016   1:00 AM            amd64
    d----         6/12/2016   1:02 AM            amd64_arm
    d----         6/12/2016   1:02 AM            amd64_x86
    d----         6/12/2016   1:02 AM            arm
    d----         6/12/2016   1:02 AM            x86_amd64
    d----         6/12/2016   1:02 AM            x86_arm
    -ar--         3/17/2016  10:48 PM     174904 atlprov.dll
    -ar--         3/17/2016  10:48 PM      92832 bscmake.exe
    -ar--         3/17/2016  10:48 PM    1182496 c1.dll
    -ar--         3/17/2016  10:48 PM    3970344 c1xx.dll
    -ar--         3/17/2016  10:48 PM    5192992 c2.dll
    -ar--         3/17/2016  10:48 PM     190096 cl.exe
    -ar--         3/17/2016  10:12 PM        409 cl.exe.config



EOU
}
vs-dir(){ echo $(local-base)/env/windows/vs/windows/vs-vs ; }
vs-cd(){  cd $(vs-dir); }
vs-mate(){ mate $(vs-dir) ; }
vs-get(){
   local dir=$(dirname $(vs-dir)) &&  mkdir -p $dir && cd $dir

}


vs-bindir(){ echo "/c/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin" ; }
vs-bincd(){ cd "$(vs-bindir)" ; }


vs-wp(){  
   echo $(vs-gitbash2win $1)
}

vs-gitbash2win(){
  local gbp=$1
  local wnp
  case $gbp in
    /c/*) wnp=${gbp//\//\\}  ;;
       *) echo expecting gitbash style path starting with /c ;;
  esac
  echo "C:${wnp:2}"
}



