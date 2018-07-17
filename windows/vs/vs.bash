# === func-gen- : windows/vs/vs fgp windows/vs/vs.bash fgn vs fgh windows/vs
vs-src(){      echo windows/vs/vs.bash ; }
vs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vs-src)} ; }
vs-vi(){       vi $(vs-source) ; }
vs-env(){      elocal- ; }
vs-usage(){ cat << \EOU


Microsoft Visual Studio 2017 Community Edition
================================================

* see also optickswin2-

* https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio

Installer offers several Workloads including::

   Windows
       Universal Windows Platform development  
       .NET desktop development
       Desktop development with C++ [6.67 GB] [PICKED JUST THIS ONE]

   Web & Cloud
       ASP.NET ..
       Azure ..
       Python .. [1.92 GB]
       Node.js
       Data storage
       Data science and analytical applications (Python, R and F#) [8.97 GB]
       Office/SharePoint ...

   Mobile & Gaming
       Mobile development with .NET
       Game development with Unity [3.91 GB]
       Mobile development with JavaScript (Android, iOS, UWP apps using Tools for Apache Cordova) [1.66 GB]
       Mobile development with C++ (cross-platform iOS, Android or Windows using C++) [10 GB]
       Game development with C++ (DirectX, Unreal or Cocos2d) [6.22 GB]

   Other Toolsets
       Visual studio extension development
       Linux development with C++ (Create and debug applications running in a Linux environment) [6.07 GB] 
       .NET Core cross-platform development


Visual Studio CMake support
-----------------------------

* https://blogs.msdn.microsoft.com/vcblog/2016/10/05/cmake-support-in-visual-studio/

Visual Studio 2017 introduces built-in support for handling CMake projects.
This makes it a lot simpler to develop C++ projects built with CMake without
the need to generate VS projects and solutions from the command line. This post
gives you an overview of the CMake support, how to easily get started and stay
productive in Visual Studio.


* https://dmerej.info/blog/post/cmake-visual-studio-and-the-command-line/





Microsoft Visual Studio 2015 Community Edition
================================================

* see also optickswin-

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


Import settings from commandline
---------------------------------

* http://stackoverflow.com/questions/21455741/export-import-visual-studio-settings-from-command-line

::

   devenv /ResetSettings c:\full\path\to\your\own.vssettings

Probably for the PATH envvar would a global setting of PATH for the VS process be honoured 
by the debug runner ?


* http://stackoverflow.com/questions/26913714/how-to-automate-changing-the-project-settings-in-visual-studio-2005-and-later

Property sheets can be added with *View -> Property Manager*



Studioshell : Powershell control of VS 
-----------------------------------------

* http://studioshell.codeplex.com/
* https://www.nuget.org/packages/StudioShell/

Provides a PS drive type interface to VS objects.

* Unfortunately does not support VS 2015.


Windows BAT scripts
---------------------

* http://stackoverflow.com/questions/5034076/what-does-dp0-mean-and-how-does-it-work


Windows envvars
-----------------

Set Via GUI
~~~~~~~~~~~~~

* Control Panel > System and Security > System > [Advanced System Settings] [Environment Variables...]
* Or just search for environment from Control Panel

* http://winaero.com/blog/how-to-see-names-and-values-of-environment-variables-in-windows-8-and-windows-7/


Usage in GUI
~~~~~~~~~~~~~

Enter in boxes surrounded by percent %NameOfEnvvar% 

Powershell
~~~~~~~~~~~

* http://stackoverflow.com/questions/23255430/how-to-change-environment-variable-powershell-and-launch-an-application

::

   get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
   gp "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment"



* https://blogs.technet.microsoft.com/heyscriptingguy/2011/07/23/use-powershell-to-modify-your-environmental-path/
* http://www.computerperformance.co.uk/powershell/powershell_env_path.htm



Resolving a command against App Paths

* http://poshcode.org/170

::

    #################################################################################################
    ## Example Usage:
    ##    Get-App Notepad
    ##       Finds notepad.exe using Get-Command
    ##    Get-App pbrush
    ##       Finds mspaint.exe using the "App Paths" registry key
    ##    &(Get-App WinWord)
    ##       Finds, and launches, Word (if it's installed) using the "App Paths" registry key
    ##################################################################################################
    ## Revision History
    ## 1.0 - initial release
    ##################################################################################################

    function Get-App {
       param( [string]$cmd )
       $eap = $ErrorActionPreference
       $ErrorActionPreference = "SilentlyContinue"
       Get-Command $cmd
       if(!$?) {
          $AppPaths = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
          if(!(Test-Path $AppPaths\$cmd)) {
             $cmd = [IO.Path]::GetFileNameWithoutExtension($cmd)
             if(!(Test-Path $AppPaths\$cmd)){
                $cmd += ".exe"
             }
          }
          if(Test-Path $AppPaths\$cmd) {
             Get-Command (Get-ItemProperty $AppPaths\$cmd)."(default)"
          }
       }
    }



App Paths registry key as alternative to PATH
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://news.ycombinator.com/item?id=10449311
* https://blogs.msdn.microsoft.com/oldnewthing/20110725-00/?p=10073/

Using Registry

* https://msdn.microsoft.com/en-us/ms997545.aspx

Application Registration

* https://msdn.microsoft.com/en-us/library/windows/desktop/ee872121#app_exe
* seems this just allows setting the PATH for an application not other envvars the process needs


* https://helgeklein.com/blog/2010/08/how-the-app-paths-registry-key-makes-windows-both-faster-and-safer/


Windows Package Manager : eg for python, ipython, numpy 
---------------------------------------------------------

Currently using 

* pacman in MSYS2/MinGW shell 


VS Natively are using:

* chocolatey- (trustability issue) 
* nuget- (trustability issue) : to investigate 

Another VS native approach maybe 

* conda/miniconda a precooked distribution and/or pkg manager
* http://conda.pydata.org/miniconda.html


Using VS GUI with A CMake Generated Solution
-----------------------------------------------

* https://cognitivewaves.wordpress.com/cmake-and-visual-studio/

* in right pane "Solution Explorer" rightclick 
  the relevant target and  "Set As Startup project" 


Hacks to modify the .sln to avoid this 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Would be better for cmake to have some way to control target order.

* http://stackoverflow.com/questions/7304625/how-do-i-change-the-startup-project-of-a-visual-studio-solution-via-cmake
* https://github.com/rpavlik/cmake-modules/blob/master/CreateLaunchers.cmake


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


Windef.h near far macros
---------------------------

Identifiers "near" and "far" (maybe "NEAR" "FAR" too) 
are not possible in code that includes Windef.h as it defines them to nothing.

* Blast from the past : **AMAZING**
* http://stackoverflow.com/questions/16814409/windef-h-why-are-far-and-near-still-here-c


Perl inplace edit drops .bak when not asked
------------------------------------------------

* http://stackoverflow.com/questions/2616865/why-do-i-have-to-specify-the-i-switch-with-a-backup-extension-when-using-active

Workaround example::

   plog-inplace-edit(){
       perl -pi -e 's,BLog\.hh,PLOG.hh,g' *.cc && rm *.cc.bak
   }



cl.exe disable warning globally rather than via pragma
----------------------------------------------------------

Powershell after (vs-export), cl -help::

    .. many pages down .. 

    /nologo suppress copyright message
    /sdl enable additional security features and warnings
    /showIncludes show include file names   /Tc<source file> compile file as .c
    /Tp<source file> compile file as .cpp   /TC compile all files as .c
    /TP compile all files as .cpp           /V<string> set version string
    /w disable all warnings                 /wd<n> disable warning n
    /we<n> treat warning n as an error      /wo<n> issue warning n once
    /w<l><n> set warning level 1-4 for n    /W<n> set warning level (default n=1)
    /Wall enable all warnings               /WL enable one line diagnostics
    /WX treat warnings as errors            /Yc[file] create .PCH file
    /Yd put debug info in every .OBJ        /Yl[sym] inject .PCH ref for debug lib
    /Yu[file] use .PCH file                 /Y- disable all PCH options





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



Missing Dependency Debugging
-------------------------------

1. open Opticks.sln into Visual Studio, from powershell with: opticks-vs 
2. right click the target with the issue, eg CPropLibTest, and "Set as Startup Project"
3. hit F5 to build and launch

A dialog box pops up saying eg "G4Tree.dll" and after the G4 path fix 
xerces-c_3_1D.dll


Windows Git Bash PATH and PATH length limitation 
---------------------------------------------------

* http://superuser.com/questions/607533/windows-git-bash-bash-path-to-read-windows-path-system-variable


Append PATH to include directory of dll
-----------------------------------------

In powershell profile (vip from powershell)::


     10
     11 $env:OPTICKS_PREFIX_OLD = "C:\Users\ntuhep\local\opticks"
     12 $env:OPTICKS_PREFIX = "C:\usr\local\opticks"
     13
     14 $env:PATH = "${env:OPTICKS_PREFIX}\lib;$env:PATH"
     15 $env:PATH = "${env:OPTICKS_PREFIX}\externals\lib;$env:PATH"
     16 $env:PATH = "${env:OPTICKS_PREFIX_OLD}\externals\bin;$env:PATH"
     17


In gitbash profile (vip from gitbash)::

     16 export OPTICKS_PREFIX_OLD=/c/Users/ntuhep/local/opticks
     17 export OPTICKS_PREFIX=/c/usr/local/opticks
     18
     19 PATH=$OPTICKS_PREFIX/lib:$PATH
     20 PATH=$OPTICKS_PREFIX/externals/lib:$PATH
     21 PATH=$OPTICKS_PREFIX_OLD/externals/bin:$PATH


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



