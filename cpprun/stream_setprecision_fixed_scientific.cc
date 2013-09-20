/*
Check default precision used in streaming doubles 

::

    simon:cpprun blyth$ stream_setprecision_fixed_scientific
    -15954.1 -805788 -4145.12,

    simon:cpprun blyth$ PRECISION=5 stream_setprecision_fixed_scientific
    -15954 -8.0579e+05 -4145.1,

    simon:cpprun blyth$ PRECISION=5 FIXED=1 stream_setprecision_fixed_scientific
    -15954.12346 -805788.12346 -4145.12346,

    simon:cpprun blyth$ PRECISION=5 SCIENTIFIC=1 stream_setprecision_fixed_scientific
    -1.59541e+04 -8.05788e+05 -4.14512e+03,

    simon:cpprun blyth$ WIDTH=5 PRECISION=3 FIXED=1 stream_setprecision_fixed_scientific   ## WIDTH has no teeth
    -15954.123 -805788.123 -4145.123,


*/
#include <fstream>  
#include <iomanip> 
#include <stdlib.h> 
#include <iostream>

int getenvvar(const char* name, int def)
{
   int ivar = def ; 
   char* evar = getenv(name);
   if (evar!=NULL) ivar = atoi(evar);
   return ivar ;
}

#define fDest std::cout

int main(int argc,char** argv)
{
   //std::ofstream fDest ;  
   //fDest.open("/tmp/stream_setprecision_fixed_scientific.out");

   double xyz[] = { -15954.123456789,-805788.123456789,-4145.123456789 };

   int prec = getenvvar("PRECISION",-1);
   int width = getenvvar("WIDTH",-1);
   int fixed = getenvvar("FIXED",-1);
   int scientific = getenvvar("SCIENTIFIC",-1);

   if(width > 0) fDest << std::setw(width) ;
   if(prec > 0)  fDest << std::setprecision(prec) ;
   if(fixed > 0) fDest << std::fixed ;     
   if(scientific > 0) fDest << std::scientific ;     

   fDest << xyz[0] << " ";
   fDest << xyz[1] << " ";
   fDest << xyz[2] << "," << "\n";

   //fDest.close() ;

   return 0 ;
}

