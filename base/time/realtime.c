//  cc realtime.c getRealTime.c -o $LOCAL_BASE/env/bin/realtime

#include "getRealTime.h"
#include "stdio.h"

int main()
{
   double t = getRealTime();
   printf("t = %f \n", t );

}
