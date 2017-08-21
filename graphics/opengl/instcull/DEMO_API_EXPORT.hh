
#pragma once

/* 

Source "Generated" hdr DEMO_API_EXPORT.hh 
Created Fri, Jun 17, 2016  8:03:44 PM with commandline::

    importlib-exports DEMO DEMO_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define DEMO_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define DEMO_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(DEMO_EXPORTS)
       #define  DEMO_API __declspec(dllexport)
   #else
       #define  DEMO_API __declspec(dllimport)
   #endif

#else

   #define DEMO_API  __attribute__ ((visibility ("default")))

#endif


