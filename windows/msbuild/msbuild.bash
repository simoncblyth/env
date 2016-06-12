# === func-gen- : windows/msbuild/msbuild fgp windows/msbuild/msbuild.bash fgn msbuild fgh windows/msbuild
msbuild-src(){      echo windows/msbuild/msbuild.bash ; }
msbuild-source(){   echo ${BASH_SOURCE:-$(env-home)/$(msbuild-src)} ; }
msbuild-vi(){       vi $(msbuild-source) ; }
msbuild-env(){      elocal- ; }
msbuild-usage(){ cat << EOU

MSBuild
=========

* https://msdn.microsoft.com/en-us/library/dd393573.aspx

The Visual Studio project system is based on MSBuild.


OME Patched xercecsc  
----------------------

See ome- xercesc-

::

packages/xerces/build.cmake::

      6 if(WIN32)
      7 
      8   message(STATUS "Building xerces (Windows)")
      9 
     10   execute_process(COMMAND msbuild "projects\\Win32\\${XERCES_SOLUTION}\\xerces-all\\xerces-all.sln"
     11                           "/p:Configuration=${XERCES_CONFIG}"
     12                           "/p:Platform=${XERCES_PLATFORM}"
     13                           "/p:useenv=true" "/v:d"
     14                   WORKING_DIRECTORY ${SOURCE_DIR}
     15                   RESULT_VARIABLE build_result)

::

       msbuild "projects\\Win32\\VC14\\xerces-all\\xerces-all.sln" "/p:Configuration=ICU Debug" "/p:Platform=x64" "/p:useenv=true" "/v:d"



packages/xerces/common.cmake::

      1 if(WIN32)
      2   set(XERCES_CONFIG "ICU Debug")
      3   if(CONFIG MATCHES "Rel")
      4     set(XERCES_CONFIG "ICU Release")
      5   endif()
      6 
      7   set(XERCES_PLATFORM Win32)
      8   if(EP_PLATFORM_BITS STREQUAL 64)
      9     set(XERCES_PLATFORM x64)
     10   endif()
     11 
     12   set(XERCES_PLATFORM_PATH Win32)
     13   if(EP_PLATFORM_BITS STREQUAL 64)
     14     set(XERCES_PLATFORM_PATH Win64)
     15   endif()
     16 
     17   # VS 12.0
     18   if(NOT MSVC_VERSION VERSION_LESS 1800 AND MSVC_VERSION VERSION_LESS 1900)
     19     set(XERCES_SOLUTION VC12)
     20   # VS 14.0
     21   elseif(NOT MSVC_VERSION VERSION_LESS 1900 AND MSVC_VERSION VERSION_LESS 2000)
     22     set(XERCES_SOLUTION VC14)
     23   else()
     24     message(FATAL_ERROR "VS version not supported by xerces")
     25   endif()
     26 endif()





SLN
----

A bunch of references to vcxproj files 

::

     .1 Microsoft Visual Studio Solution File, Format Version 12.00
      2 # Visual Studio 14
      3 VisualStudioVersion = 14.0.25123.0
      4 MinimumVisualStudioVersion = 10.0.40219.1
      5 Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "all", "all\all.vcxproj", "{E305E46C-9D74-4755-BF57-29DEAEF4DCDD}"
      6 EndProject
      7 Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "CreateDOMDocument", "CreateDOMDocument\CreateDOMDocument.vcxproj", "{8709DC2A-0EC9-4B67-9E98-95D5174B2C3F}"
      8 EndProject
      9 Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "DOMCount", "DOMCount\DOMCount.vcxproj", "{60E3008A-0D78-4B25-A12E-9D7A3921F67C}"
     ...
     53 Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "XercesLib", "XercesLib\XercesLib.vcxproj", "{152CE948-F659-4206-A50A-1D2B9658EF96}"
     54 EndProject



ICU is a configure option, to use library: International Components for Unicode

* https://xerces.apache.org/xerces-c/build-3.html


::

     68     GlobalSection(SolutionConfigurationPlatforms) = preSolution
     69         Debug|Win32 = Debug|Win32
     70         Debug|x64 = Debug|x64
     71         ICU Debug|Win32 = ICU Debug|Win32
     72         ICU Debug|x64 = ICU Debug|x64
     73         ICU Release|Win32 = ICU Release|Win32
     74         ICU Release|x64 = ICU Release|x64
     75         Release|Win32 = Release|Win32
     76         Release|x64 = Release|x64
     77         Static Debug|Win32 = Static Debug|Win32
     78         Static Debug|x64 = Static Debug|x64
     79         Static Release|Win32 = Static Release|Win32
     80         Static Release|x64 = Static Release|x64
     81     EndGlobalSection
     82     GlobalSection(ProjectConfigurationPlatforms) = postSolution
     83         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.Debug|Win32.ActiveCfg = Debug|Win32
     84         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.Debug|Win32.Build.0 = Debug|Win32
     85         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.Debug|x64.ActiveCfg = Debug|x64
     86         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.Debug|x64.Build.0 = Debug|x64
     87         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.ICU Debug|Win32.ActiveCfg = ICU Debug|Win32
     88         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.ICU Debug|Win32.Build.0 = ICU Debug|Win32
     89         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.ICU Debug|x64.ActiveCfg = ICU Debug|x64
     90         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.ICU Debug|x64.Build.0 = ICU Debug|x64
     91         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.ICU Release|Win32.ActiveCfg = ICU Release|Win32
     92         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.ICU Release|Win32.Build.0 = ICU Release|Win32
     93         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.ICU Release|x64.ActiveCfg = ICU Release|x64
     94         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.ICU Release|x64.Build.0 = ICU Release|x64
     95         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.Release|Win32.ActiveCfg = Release|Win32
     96         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.Release|Win32.Build.0 = Release|Win32
     97         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.Release|x64.ActiveCfg = Release|x64
     98         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.Release|x64.Build.0 = Release|x64
     99         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.Static Debug|Win32.ActiveCfg = Static Debug|Win32
    100         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.Static Debug|Win32.Build.0 = Static Debug|Win32
    101         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.Static Debug|x64.ActiveCfg = Static Debug|x64
    102         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.Static Debug|x64.Build.0 = Static Debug|x64
    103         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.Static Release|Win32.ActiveCfg = Static Release|Win32
    104         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.Static Release|Win32.Build.0 = Static Release|Win32
    105         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.Static Release|x64.ActiveCfg = Static Release|x64
    106         {E305E46C-9D74-4755-BF57-29DEAEF4DCDD}.Static Release|x64.Build.0 = Static Release|x64
    ...
    659         {152CE948-F659-4206-A50A-1D2B9658EF96}.Debug|Win32.ActiveCfg = Debug|Win32
    660         {152CE948-F659-4206-A50A-1D2B9658EF96}.Debug|Win32.Build.0 = Debug|Win32
    661         {152CE948-F659-4206-A50A-1D2B9658EF96}.Debug|x64.ActiveCfg = Debug|x64
    662         {152CE948-F659-4206-A50A-1D2B9658EF96}.Debug|x64.Build.0 = Debug|x64
    663         {152CE948-F659-4206-A50A-1D2B9658EF96}.ICU Debug|Win32.ActiveCfg = ICU Debug|Win32
    664         {152CE948-F659-4206-A50A-1D2B9658EF96}.ICU Debug|Win32.Build.0 = ICU Debug|Win32
    665         {152CE948-F659-4206-A50A-1D2B9658EF96}.ICU Debug|x64.ActiveCfg = ICU Debug|x64
    666         {152CE948-F659-4206-A50A-1D2B9658EF96}.ICU Debug|x64.Build.0 = ICU Debug|x64
    667         {152CE948-F659-4206-A50A-1D2B9658EF96}.ICU Release|Win32.ActiveCfg = ICU Release|Win32
    668         {152CE948-F659-4206-A50A-1D2B9658EF96}.ICU Release|Win32.Build.0 = ICU Release|Win32
    669         {152CE948-F659-4206-A50A-1D2B9658EF96}.ICU Release|x64.ActiveCfg = ICU Release|x64
    670         {152CE948-F659-4206-A50A-1D2B9658EF96}.ICU Release|x64.Build.0 = ICU Release|x64
    671         {152CE948-F659-4206-A50A-1D2B9658EF96}.Release|Win32.ActiveCfg = Release|Win32
    672         {152CE948-F659-4206-A50A-1D2B9658EF96}.Release|Win32.Build.0 = Release|Win32
    673         {152CE948-F659-4206-A50A-1D2B9658EF96}.Release|x64.ActiveCfg = Release|x64
    674         {152CE948-F659-4206-A50A-1D2B9658EF96}.Release|x64.Build.0 = Release|x64
    675         {152CE948-F659-4206-A50A-1D2B9658EF96}.Static Debug|Win32.ActiveCfg = Static Debug|Win32
    676         {152CE948-F659-4206-A50A-1D2B9658EF96}.Static Debug|Win32.Build.0 = Static Debug|Win32
    677         {152CE948-F659-4206-A50A-1D2B9658EF96}.Static Debug|x64.ActiveCfg = Static Debug|x64
    678         {152CE948-F659-4206-A50A-1D2B9658EF96}.Static Debug|x64.Build.0 = Static Debug|x64
    679         {152CE948-F659-4206-A50A-1D2B9658EF96}.Static Release|Win32.ActiveCfg = Static Release|Win32
    680         {152CE948-F659-4206-A50A-1D2B9658EF96}.Static Release|Win32.Build.0 = Static Release|Win32
    681         {152CE948-F659-4206-A50A-1D2B9658EF96}.Static Release|x64.ActiveCfg = Static Release|x64
    682         {152CE948-F659-4206-A50A-1D2B9658EF96}.Static Release|x64.Build.0 = Static Release|x64




EOU
}
msbuild-dir(){ echo $(local-base)/env/windows/msbuild/windows/msbuild-msbuild ; }
msbuild-cd(){  cd $(msbuild-dir); }
msbuild-mate(){ mate $(msbuild-dir) ; }
msbuild-get(){
   local dir=$(dirname $(msbuild-dir)) &&  mkdir -p $dir && cd $dir

}
