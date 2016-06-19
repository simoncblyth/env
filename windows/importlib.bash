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



Can you dllexport/dllimport an inline function?
-------------------------------------------------

* https://blogs.msdn.microsoft.com/oldnewthing/20140109-00/?p=2123

My take: **avoid inlining as much as possible**


dllexport of class with std:vector<int> base 
----------------------------------------------------

* http://www.codesynthesis.com/~boris/blog/2010/01/18/dll-export-cxx-templates/

Implicit exports of base class::

    // lib 

    class BASE_EXPORT ints: public std::vector<int>
    {
      ...
    };

Can cause duplicated symbols in client code that uses that base already::

    void f ()
    {
         std::vector<int> v;
 
         ints i();  

         ...
    }


Workaround is to be explicit in client code::

   template class __declspec(dllimport) std::vector<int>;


HowTo: Export C++ classes from a DLL
--------------------------------------

* http://www.codeproject.com/Articles/28969/HowTo-Export-C-classes-from-a-DLL

* C Language Approach
* C++ Naive Approach: Exporting a Class
* C++ Mature Approach: Using an Abstract Interface

I am using "Naive" approach as doing otherwise would be a major rewrite.
The result is that have to use same compiler (and flags) for everything. 

Elaboration on *same compiler and flags*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below comment on above article elaborates on *same compiler and flags*

* http://stackoverflow.com/questions/16663794/how-to-export-c-class-as-a-dll

Note that if you build a DLL with C++ classes at the boundaries (including MFC
classes, or STL classes), your DLL client must use the same VC++ compiler
version and the same CRT flavor (e.g. multithreaded DLL debug CRT,
multithreaded DLL release CRT, and other "more subtle" settings, e.g. same
_HAS_ITERATOR_DEBUGGING settings) to build the EXEs that will use the DLL.

Instead, if you export a pure C interface from the DLL (but you can use C++
inside the DLL, just like Win32 APIs), or if you build COM DLLs, your DLL
clients can use a different version of VC++ compiler (and even different CRTs)
to use your DLL.


Abstract interface example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://eli.thegreenplace.net/2011/09/16/exporting-c-classes-from-a-dll

Pimpl vs Pure Virtual
~~~~~~~~~~~~~~~~~~~~~~~~

Does pimpl hide enough or does it really need to be pure virtual base ?

* https://www.reddit.com/r/gamedev/comments/1kaky3/workaround_to_exporting_classes_to_a_dll_with_c/
* http://bitsquid.blogspot.tw/2012/03/pimpl-vs-pure-virtual-interfaces.html


* http://stackoverflow.com/questions/26290372/dllexport-pure-virtual-class-with-private-implementation-not-working

Is the end statement "Hence your implementation must be exported" correct ? TODO: ask the compiler.


dllexport templated member function
--------------------------------------

* http://stackoverflow.com/questions/1969343/cannot-export-template-function

Suggests to implemnet the template method in header rather than imp.

Export the template specialization::

    template __declspec(dllexport) void updateParamValue<int>( const std::string& name, T val );
    template __declspec(dllexport) void updateParamValue<short>( const std::string& name, T val );
    ......




* https://anteru.net/blog/2008/11/19/318/

Split header approach.


* http://stackoverflow.com/questions/27090975/c-template-specialization-in-different-dll-produces-linker-errors

Suggests it might be possible to have a separte imp with explicit instantiation.


MSVC : General Rules and Limitations
--------------------------------------

* https://msdn.microsoft.com/en-us/library/twa2aw10(VS.80).aspx

..if you apply dllexport to a regular class that has a base class that is not marked as dllexport, the compiler will generate C4275.

The compiler generates the same warning if the base class is a specialization
of a class template. To work around this, mark the base-class with dllexport.
The problem with a specialization of a class template is where to place the
__declspec(dllexport); **you are not allowed to mark the class template**. 

*Is this saying the compiler doesnt accept double export*
::

   class __declspec(dllexport) D : public __declspec(dllexport) B<int> 
 

Instead, explicitly instantiate the class template and 
mark this explicit instantiation with dllexport. 
For example::

    template class __declspec(dllexport) B<int>;

    class __declspec(dllexport) D : public B<int> 
    {
       // ...


This workaround fails if the template argument is the deriving class. For example::

    class __declspec(dllexport) D : public B<D> {
    // ...


Because this is common pattern with templates, the compiler changed the
semantics of dllexport when it is applied to a class that has one or more
base-classes and when one or more of the base classes is a specialization of a
class template. In this case, the compiler implicitly applies dllexport to the
specializations of class templates. In Visual C++ .NET, a user can do the
following and not get a warning:

::

    class __declspec(dllexport) D : public B<D> {
    // ...



MSVC : Differences from Other Implementations
--------------------------------------------------------

* https://msdn.microsoft.com/en-us/library/0y86hzch.aspx

* The compiler cannot instantiate a template outside of the module in which it is defined. 
  Visual C++ does not support the export keyword.

  **does this mean cannot do explict instantiation in the .cpp that keeps headers cleaner?**

* Templates cannot be used with functions declared with __declspec (dllimport) or __declspec (dllexport).

  **seems not to be true for fully specialized templates** 

* All template arguments must be of an unambiguous type that exactly matches that of the template parameter list. 
  For example:

     template< class T > T check( T );
     template< class S > void watch( int (*)(S) );
     watch( check );     //error

  The compiler should instantiate the check templated function in the form *int check( int )*, 
  but the inference cannot be followed.

* When resolving names used in class templates or function templates, all names are treated as dependent names. 
  See Name Resolution for Dependent Types

* In a class template, the template parameter can be redefined in the scope of the class definition. 
  See Name Resolution for Locally Declared Names




BFoo : Exporting a templated function
----------------------------------------

::

      1 #include <iostream>
      2
      3 #include "BRAP_API_EXPORT.hh"
      4 #include "BRAP_FLAGS.hh"
      5
      6 template<typename T>
      7 BRAP_API void foo(T value)
      8 {
      9     std::cerr << "BFoo"
     10               << " value " << value
     11               << std::endl
     12               ;
     13 }
     14
     15 template BRAP_API void foo<int>(int);
     16 template BRAP_API void foo<double>(double);
     17 template BRAP_API void foo<char*>(char*);
     18

::

      C:\Users\ntuhep\env\boostrap\BFoo.hh(13): error C2491: 'foo': 
      definition of dllimport function not allowed 
      [C:\usr\local\opticks\build\boostrap\tests\BFooTest.vcxproj]



Fix by removing the export/impot from the template just apply to specialization::

      6 template<typename T>
      7 void foo(T value)
      8 {
 


* https://social.msdn.microsoft.com/Forums/vstudio/en-US/4fd49664-e28e-4f23-b1eb-b669d35ad264/function-template-instantation-export-from-dll?forum=vcgeneral

* https://social.msdn.microsoft.com/Forums/vstudio/en-US/0d613b65-52ac-4fb7-bf65-8a543dfbcc6e/visual-c-error-lnk2019-unresolved-external-symbol?forum=vcgeneral






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

 
