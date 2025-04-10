
#pragma once

/* 

Source "Generated" hdr OKGL_API_EXPORT.hh 
Created Sat Jun 25 14:18:57 CST 2016 with commandline::

    importlib-exports OpticksGL OKGL_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define OpticksGL_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define OpticksGL_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(OpticksGL_EXPORTS)
       #define  OKGL_API __declspec(dllexport)
   #else
       #define  OKGL_API __declspec(dllimport)
   #endif

#else

   #define OKGL_API  __attribute__ ((visibility ("default")))

#endif


