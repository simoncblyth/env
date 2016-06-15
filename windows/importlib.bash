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



