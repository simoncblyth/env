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



Exporting STL 
---------------------

* http://stackoverflow.com/questions/4145605/stdvector-needs-to-have-dll-interface-to-be-used-by-clients-of-class-xt-war

Exporting from a DLL is platform-specific. You will have to fix this for
Windows (basically use declspec(dllexport/dllimport) on the instantiated class
template) and encapsulate the required code in your Windows-specific
preprocessor macro.

My experience is that exporting STL classes from DLLs on Windows is fraught
with pain, generally I try to design the interface such that this is not
needed.

::

    (ClCompile target) ->
      c:\users\ntuhep\env\numerics\npy\NPYBase.hpp(142): warning C4251: 'NPYBase::m_shape': class 'std::vector<int,std::allocator<_Ty>>' needs to have dll-interface to be used by clients of class 'NPYBase' [C:\usr\local\opticks\build\numerics\npy\NPY.vcxproj]
      c:\users\ntuhep\env\numerics\npy\NPYBase.hpp(143): warning C4251: 'NPYBase::m_metadata': class 'std::basic_string<char,std::char_traits<char>,std::allocator<char>>' needs to have dll-interface to be used by clients of class 'NPYBase' [C:\usr\local\opticks\build\numerics\npy\NPY.vcxproj]
      c:\users\ntuhep\env\numerics\npy\NPY.hpp(199): warning C4251: 'NPY<T>::m_data': class 'std::vector<Scalar,std::allocator<_Ty>>' needs to have dll-interface to be used by clients of class 'NPY<T>' [C:\usr\local\opticks\build\numerics\npy\NPY.vcxproj]





sharedLibsDemo
-----------------

msvc build
~~~~~~~~~~~~~~~


Compile shared.cpp into shared.obj with compilation flag shared_EXPORTS::

       C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\CL.exe /c
       /I"C:\usr\local\env\windows\sharedLibsDemo\build-windows-msvc" 
       /IC:\usr\local\env\windows\sharedLibsDemo 
       /nologo /W3 /WX- /O2 /Ob2 /Oy- 
       /D WIN32 /D _WINDOWS /D NDEBUG /D "CMAKE_INTDIR=\"Release\"" 

       /D shared_EXPORTS 

       /D _WINDLL /D _MBCS 
       /Gm- /EHsc /MD /GS /fp:precise /Zc:wchar_t /Zc:forScope /Zc:inline 
       /GR 
       /Fo"shared.dir\Release\\" 
       /Fd"shared.dir\Release\vc140.pdb" 
       /Gd /TP /analyze- /errorReport:queue 
       C:\usr\local\env\windows\sharedLibsDemo\shared.cpp

Link shared.obj into shared.dll AND IMPLIB shared.lib::

       C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\link.exe
       /ERRORREPORT:QUEUE
       /OUT:"C:\usr\local\env\windows\sharedLibsDemo\build-windows-msvc\Release\shared.dll" 
       /INCREMENTAL:NO /NOLOGO 
       kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib  
       /MANIFEST
       /MANIFESTUAC:"level='asInvoker' uiAccess='false'" 
       /manifest:embed
       /PDB:"C:/usr/local/env/windows/sharedLibsDemo/build-windows-msvc/Release/shared.pdb" 
       /SUBSYSTEM:CONSOLE /TLBID:1 /DYNAMICBASE /NXCOMPAT
       /IMPLIB:"C:/usr/local/env/windows/sharedLibsDemo/build-windows-msvc/Release/shared.lib"
       /MACHINE:X86 /SAFESEH  /machine:X86 
       /DLL
       shared.dir\Release\shared.obj

Compile main.cpp into main.obj WITHOUT compilation flag shared_EXPORTS::

       ClCompile: C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\CL.exe /c
       /I"C:\usr\local\env\windows\sharedLibsDemo\build-windows-msvc"
       /IC:\usr\local\env\windows\sharedLibsDemo
       /nologo /W3 /WX- /O2 /Ob2 /Oy- 

       /D WIN32 /D _WINDOWS /D NDEBUG /D "CMAKE_INTDIR=\"Release\"" /D _MBCS 

       /Gm- /EHsc /MD /GS /fp:precise /Zc:wchar_t /Zc:forScope /Zc:inline /GR 
       /Fo"main.dir\Release\\"
       /Fd"main.dir\Release\vc140.pdb" 
       /Gd /TP /analyze- /errorReport:queue
       C:\usr\local\env\windows\sharedLibsDemo\main.cpp 


Link main.obj and shared.lib into main.exe. 

* main.lib is mentioned but not created
* dll not mentioned,  that is discovered at runtime

::

  C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\link.exe
      /ERRORREPORT:QUEUE
      /OUT:"C:\usr\local\env\windows\sharedLibsDemo\build-windows-msvc\Release\main.exe" 
      /INCREMENTAL:NO /NOLOGO 

      kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib 
      Release\shared.lib 

      /MANIFEST /MANIFESTUAC:"level='asInvoker' uiAccess='false'"
      /manifest:embed
      /PDB:"C:/usr/local/env/windows/sharedLibsDemo/build-windows-msvc/Release/main.pdb" 
      /SUBSYSTEM:CONSOLE 
      /TLBID:1 
      /DYNAMICBASE 
      /NXCOMPAT
      /IMPLIB:"C:/usr/local/env/windows/sharedLibsDemo/build-windows-msvc/Release/main.lib"
      /MACHINE:X86 
      /SAFESEH  
      /machine:X86 main.dir\Release\main.obj



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


importlib-exports-(){ 

   local lib=${1:-MyLibrary}
   local api=${2:-MYLIB_API} 
   local hdr=$(importlib-hdr $api)

   cat << EOX

#pragma once

/* 

Source "Generated" hdr $hdr 
Created $(date) with commandline::

    importlib-exports $lib $api  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define ${lib}_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define ${lib}_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(${lib}_EXPORTS)
       #define  $api __declspec(dllexport)
   #else
       #define  $api __declspec(dllimport)
   #endif

#else

   #define $api

#endif


EOX
}


importlib-hdr(){  echo ${1}_EXPORT.hh ; }

importlib-exports(){

   local msg=" === $FUNCNAME "
   local lib=${1:-MyLibrary}
   local api=${2:-MYLIB_API} 
   local hdr=$(importlib-hdr $api)

   echo $msg lib $lib api $api hdr $hdr : generating header in PWD $PWD

   importlib-exports- $lib $api > $hdr

   cat $hdr

   echo $msg use the header in public API classes as indicated:   

   importlib-example $lib $api

}


importlib-example(){ 

   local lib=${1:-MyLibrary}
   local api=${2:-MYLIB_API} 
   local hdr=$(importlib-hdr $api)

   cat << EOX

#include "$hdr"

class $api Example 
{
   public:
       static $api void MyExampleStaticFunc();

};

EOX
}


importlib-libdir(){ echo $(importlib-dir)/build-windows-msvc/Release ; }
importlib-libdirwin(){ echo $(vs-gitbash2win $(importlib-libdir)) ; }

importlib-lib(){ echo $(importlib-find lib) ;}
importlib-dll(){ echo $(importlib-find dll) ;}
importlib-find(){
   local ext=${1:-lib}
   local iwd
   importlib-cd 
   local lib=$(find $PWD  -name "*.$ext")
   cd $iwd
   echo $lib
}


importlib-include-dirs(){
   local h
   for h in $(importlib-find h) ; do 
      echo $(dirname $h)
   done
}

 
