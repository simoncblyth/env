#include <stdio.h>
#include <stdlib.h>
#include "private.h"

int main(int argc, char** argv)
{
	int rc ;
	rc = private_init(); if(rc != EXIT_SUCCESS) exit(rc) ;
	int a ; for ( a = 1; a < argc; a++ ) printf("%s\n", private_lookup(argv[a])) ;
	rc = private_cleanup(); if(rc != EXIT_SUCCESS) exit(rc) ;
	return EXIT_SUCCESS;
}

