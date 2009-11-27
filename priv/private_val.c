#include <stdio.h>
#include <stdlib.h>
#include "private.h"

int main(int argc, char** argv)
{

        const size_t max = 100 ;
        char stamp[max] ;
        const char* afmt = "%s %s" ;
        const char* tfmt1 = "%c" ;
        private_gethostftime( stamp , max , tfmt1 , afmt  );
        printf("private_gethostftime( stamp , max , \"%s\", \"%s\" ) --> \"%s\" \n", tfmt1 , afmt , stamp );

        const char* tfmt2 = "%Y%m%d_%H%M%S" ;
        private_gethostftime( stamp , max , tfmt2 , afmt  );
        printf("private_gethostftime( stamp , max , \"%s\", \"%s\" ) --> \"%s\" \n", tfmt2 , afmt , stamp );


	int rc ;
	rc = private_init(); if(rc != EXIT_SUCCESS) exit(rc) ;
	int a ; 
        char* def = "default" ;
        for ( a = 1; a < argc; a++ ){
            char* arg = argv[a] ;
            printf("private_lookup(\"%s\") = \"%s\"\n", arg, private_lookup(arg)) ;
            printf("private_lookup_default(\"%s\",\"%s\") = \"%s\" \n", arg, def, private_lookup_default(arg,def ) ) ;
        }
	rc = private_cleanup(); if(rc != EXIT_SUCCESS) exit(rc) ;
	return EXIT_SUCCESS;
}

