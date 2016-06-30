
#pragma once

#if defined (_WIN32) 

   #if defined(omc_EXPORTS)
       #define  OMC_API __declspec(dllexport)
   #else
       #define  OMC_API __declspec(dllimport)
   #endif

#else

   #define OMC_API  __attribute__ ((visibility ("default")))

#endif

