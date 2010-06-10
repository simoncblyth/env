#include <stdio.h>
#include <stdlib.h>
#include "private.h"

int main(int argc, char** argv)
{

        const size_t max = 100 ;
        char stamp[max] ;
        const char* afmt = "%s@%s %s" ;
        const char* tfmt1 = "%c" ;
        private_getuserhostftime( stamp , max , tfmt1 , afmt  );
        printf("private_getuserhostftime( stamp , max , \"%s\", \"%s\" ) --> \"%s\" \n", tfmt1 , afmt , stamp );

        const char* tfmt2 = "%Y%m%d_%H%M%S" ;
        private_getuserhostftime( stamp , max , tfmt2 , afmt  );
        printf("private_getuserhostftime( stamp , max , \"%s\", \"%s\" ) --> \"%s\" \n", tfmt2 , afmt , stamp );

        printf( "private_hostname : %s \n", private_hostname() );
        printf( "private_username : %s \n", private_username() );
        printf( "private_userhost : %s \n", private_userhost() );

	int rc ;
	rc = private_init(); if(rc != EXIT_SUCCESS) exit(rc) ;
	int a ; 
        char* def = "AMQP_PORT" ;
        for ( a = 1; a < argc; a++ ){
            char* arg = argv[a] ;
            printf("private_lookup(\"%s\") = \"%s\"\n", arg, private_lookup(arg)) ;
            printf("private_lookup_default(\"%s\",\"%s\") = \"%s\" \n", arg, def, private_lookup_default(arg,def ) ) ;
        }
	rc = private_cleanup(); if(rc != EXIT_SUCCESS) exit(rc) ;
	return EXIT_SUCCESS;
}

