# === func-gen- : windows/nuget fgp windows/nuget.bash fgn nuget fgh windows
nuget-src(){      echo windows/nuget.bash ; }
nuget-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nuget-src)} ; }
nuget-vi(){       vi $(nuget-source) ; }
nuget-env(){      elocal- ; }
nuget-usage(){ cat << EOU

NuGet
======

* https://www.nuget.org
* https://docs.nuget.org/consume/installing-nuget

Started as .NET package manager working inside Visual Studio,
but now also operates with C++ pkgs and from commandline.

Install
--------

* https://dist.nuget.org/index.html

The choco version not far behind latest.

NuGet.CommandLine 3.4.3

* https://chocolatey.org/packages/NuGet.CommandLine

From elevated powershell::

   choco install nuget.commandline

Refs
-----

* https://docs.nuget.org/consume/nuget-faq

Commandline Usage
--------------------

* http://blog.davidebbo.com/2011/01/installing-nuget-packages-directly-from.html
* https://docs.nuget.org/consume/command-line-reference

::

    md \Check
    cd \Check
    nuget list boost*vc140
    nuget install boost-vc140


Available Binary Packages
---------------------------

Xerces-C++ XML Parser 3.1.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.nuget.org/packages/xercesc/

A single release, a few hundred downloads over a few years, no history, no supporting evidence,
no version info other than "3.1.1".


::

   md C:\Check2
   cd C:\Check
   nuget install xercesc

   # build/native/include
   # build/native/lib 2*2*2 flavors [Win32,x64]*[v100,v110]*[Debug,Release]


boost 
~~~~~~~

Multiple releases, several years history, 12k downloads.

* https://www.nuget.org/packages/boost-vc140/
* https://www.nuget.org/packages/boost/
* https://www.nuget.org/packages?q=Tags%3A%22vc140%22

Author of Boost NuGet pkgs :  Sergey Shandar, Microsoft

* http://getboost.codeplex.com
* http://getboost.codeplex.com/discussions


* http://stackoverflow.com/questions/29952757/vs2013-boost-using-nuget

Sergey::

    The structure of Boost NuGet packages is here. You can use boost-vc120 but it
    will download and link ALL boost libraries to your project. So if you want to
    save disk space, then use boost package which gives you header files, and
    specific binary packages. In your case, it's boost_system-vc120.

    You can't remove boost package because binary packages, such as
    boost-vc120/boost_system-vc120, depend on it.


Doing::

   nuget install boost-vc140

Installs 124 .dll (31*4 flavors 32,32gd,64,64gd) into that many directories.
The installer downloads .nupkg files (which are zip archive like) and extracts
them into the pwd. 


boost binary alternative 
~~~~~~~~~~~~~~~~~~~~~~~~~~

Official windows boost binaries.

* http://getboost.codeplex.com/discussions/471875
* https://sourceforge.net/projects/boost/files/boost-binaries/1.61.0/

::

    boost_1_61_0-msvc-14.0-64.exe   2016-05-08  283.4 MB


zlib
~~~~~

* https://www.nuget.org/packages/zlib/


EOU
}

nuget-open(){ open https://www.nuget.org ; }
nuget-dir(){ echo $(local-base)/env/windows/windows-nuget ; }
nuget-cd(){  cd $(nuget-dir); }
nuget-mate(){ mate $(nuget-dir) ; }
nuget-get(){
   local dir=$(dirname $(nuget-dir)) &&  mkdir -p $dir && cd $dir

}
