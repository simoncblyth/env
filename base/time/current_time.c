//  clang current_time.c -o $LOCAL_BASE/env/bin/current_time && current_time

#include <stdio.h>  
#include <time.h>   

void current_time(char* buf, int buflen, const char* tfmt, int utc)
{
   time_t t;
   time (&t); 
   struct tm* tt = utc ? gmtime(&t) : localtime(&t) ;
   strftime(buf, buflen, tfmt, tt);
}


int main ()
{
  const int buflen = 80 ;
  char buf[buflen];

  const char* tfmt = "[%s] : %Y-%m-%d %H:%M:%S" ;

  current_time(buf, buflen, tfmt, 1 );
  printf("current_time utc   %s\n", buf ); 

  current_time(buf, buflen, tfmt, 0 );
  printf("current_time local %s\n", buf ); 

  return 0;
}
