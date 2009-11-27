
#include <stdio.h>
#include <time.h>

int getftime( char* buffer , size_t max ,  const char* tfmt )
{ 
  time_t rawtime;
  time ( &rawtime );
  struct tm * timeinfo;
  timeinfo = localtime ( &rawtime );
  return strftime ( buffer, max, tfmt ,timeinfo);
}

int gethostftime( char* buffer , size_t max , const char* tfmt )
{
  const size_t tmax = 80 ;
  char tbuf[tmax] ;
  int trc = getftime( tbuf , tmax , tfmt );
  const size_t hmax = 80 ;
  char hbuf[hmax] ;
  int hrc = gethostname( hbuf , hmax );
  snprintf( buffer, max , "%s %s\n", tbuf , hbuf );
}

int main()
{
   const size_t max = 80 ;
   char buf[max] ;
   gethostftime( buf , max , "%c" ) ;
   printf( buf );
   return 0;
}



