#include "md5digest.h"

/*
[ "$(./a.out hello)" == "$(echo -n hello | md5)" ] && echo y 
*/

int main(int argc, char* argv[])
{
   for(int n=1 ; n < argc ; ++n )
   {
      const char* arg = argv[n] ;
      char* out = md5digest_str2md5( arg, strlen(arg) );
      printf("%s\n", out );
      free(out);
   }
}

