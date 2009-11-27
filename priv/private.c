/*

    Build as executable with :
        gcc private.c -Wall  -I/opt/local/include -I/opt/local/include/glib-2.0 -I/opt/local/lib/glib-2.0/include  -L/opt/local/lib -lglib-2.0 -lpcre -o private
      
   
    OSX .dylib creation and usage  (for OSX compling/linking tips : http://www.finkproject.org/doc/porting/shared.php?phpLang=en  )

        gcc -fno-common -c  private.c -Wall -I/opt/local/include -I/opt/local/include/glib-2.0 -I/opt/local/lib/glib-2.0/include 
        gcc -dynamiclib -Wl,-undefined -Wl,dynamic_lookup  -L/opt/local/lib -lglib-2.0 -lpcre  -o libprivate.dylib  private.o 

        gcc -fno-common -c private_val.c -Wall -I.     
        gcc private_val.o  -L. -lprivate -o private_val

        ./private_val AMQP_SERVER
        DYLD_LIBRARY_PATH=~/e/pcre  ~/e/pcre/private_val AMQP_SERVER 

    Linux .so
       gcc -shared private.o -L/opt/local/lib -lglib-2.0 -lpcre  -o libprivate.so

       ld private_test.o  -lSystem -L. -lprivate -isysroot /Developer/SDKs/MacOSX10.5.sdk -o private_test   
      
    based on 
          pcredemo.c   from the pcre distribution 
          hashtable article from : http://www.ibm.com/developerworks/linux/library/l-glib2.html  

*/

#include <stdio.h>
#include <stdlib.h>
#include <pcre.h>
#include <glib.h>
#include <string.h>
#include <time.h>

// gethostname
#include <unistd.h>
 
#include "private.h"

static GHashTable *table;

static void dump_hash_table_entry(gpointer key, gpointer value, gpointer user_data)
{
    printf("dump_hash_table_entry :  \"%s\" \"%s\" \n", (char*)key, (char*)value);
}

int parse_config( const char* path ){

    const char *errstr ;
    int erroff;
    pcre *re;
    const char *patn = PATTERN ;
    if (!(re = pcre_compile(patn, 0, &errstr, &erroff, 0))) {
       fprintf(stderr, "ABORT : cannot compile pattern %s: %s\n", patn, errstr );
	   return EXIT_FAILURE ;
    }

    int namecount;
    unsigned char *name_table;
    int name_entry_size;
    unsigned char *tabptr;

    (void)pcre_fullinfo(re, NULL, PCRE_INFO_NAMECOUNT, &namecount);  // number of named substrings
    if (namecount <= 0)
    {
        fprintf(stderr,"No named substrings %d \n", namecount ); 
        return EXIT_FAILURE ;
    }

    /* Before we can access the substrings, we must extract the table for
	   translating names to numbers, and the size of each entry in the table. */
    (void)pcre_fullinfo(re, NULL, PCRE_INFO_NAMETABLE, &name_table);          
    (void)pcre_fullinfo(re, NULL, PCRE_INFO_NAMEENTRYSIZE, &name_entry_size);

    FILE *file = fopen(path, "r");
    if(!file){
        printf("ABORT : file \"%s\" cannot be opened \n", path );
        return EXIT_FAILURE ;	
    }

    int ovector[OVECCOUNT];
    int rc, i ;

    char token[100] ;  
    char value[512] ; 
    char* k ;
    char* v ; 

    char line[512];  
    while ( fgets(line, sizeof line, file) ){
        size_t line_len = strcspn(line, "\n");
        rc = pcre_exec( re,  NULL, line, line_len, 0, 0, ovector,  OVECCOUNT); 
	if (rc <= 0)
	{
            switch(rc)
            {
                case 0:printf("Too many matches for ovector \n"); break;
                case PCRE_ERROR_NOMATCH: /*printf("No match\n");*/ break;
                default: printf("Matching error %d\n", rc); break;
            } 
        }
	else 
	{
            // try to grab the k,v pair from the named substrings matched on this line ...
            k = NULL ;
            v = NULL ;	 
            tabptr = name_table;
            for (i = 0; i < namecount; i++)
            {
                int n = (tabptr[0] << 8) | tabptr[1];	 
                sprintf( token, "%*s", name_entry_size - 3, tabptr + 2 );
                sprintf( value, "%.*s", ovector[2*n+1] - ovector[2*n], line + ovector[2*n] );
                if(strcmp(token, " pname") == 0 ) k = g_strdup(value) ; // WHY THE LEADING SPACE IN THE TOKEN ?
                if(strcmp(token, "pvalue") == 0)  v = g_strdup(value) ;
		tabptr += name_entry_size;
            }                                   // over named substrings
				     
            if( k && v ){
                char *old_k, *old_v;
                //  replace and free if preexisting key ... otherwise just insert into hashtable  
		if(g_hash_table_lookup_extended(table, k, (gpointer*)&old_k, (gpointer*)&old_v)) {
                    g_hash_table_insert(table, g_strdup(k), g_strdup(v));
                    g_free(old_k);
                    g_free(old_v);
                } else {
                    g_hash_table_insert(table, g_strdup(k), g_strdup(v));
		}
            } else {
                printf("failed to pluck both k and v ... skipping the line  ") ;
	    }	
        }      // a matched line
  }            // over lines of the file
  return EXIT_SUCCESS ;
}

int private_init()
{
    const char* epp = ENVVAR ; 
    char* path = getenv(epp);
    if(!path){
	printf("ABORT : envvar %s is not defined \n", epp );
	return EXIT_FAILURE;
    }
    table = g_hash_table_new(g_str_hash, g_str_equal);  // funcs for : hashing, key comparison 
    if(parse_config( path ) != 0){
        printf("ABORT: error during parse_config\n");
        return EXIT_FAILURE ;
    }
    return EXIT_SUCCESS ;
}

const char* private_lookup( const char* key )
{
    const char* value = NULL ;
    value = g_hash_table_lookup(table, key);  // returns NULL if not found 
    return value ;
}

const char* private_lookup_default( const char* key , const char* def )
{
    const char* value = private_lookup( key );
    return value ? value : def  ;
}

int private_cleanup()
{
    g_hash_table_destroy(table);
    return EXIT_SUCCESS;
}

int private_dump()
{
    g_hash_table_foreach(table, dump_hash_table_entry, NULL);
    return EXIT_SUCCESS ;
}

int private_getftime( char* buffer , size_t max ,  const char* tfmt )
{ 
  time_t rawtime;
  time ( &rawtime );
  struct tm * timeinfo;
  timeinfo = localtime ( &rawtime );
  return strftime ( buffer, max, tfmt ,timeinfo);
}

int private_gethostftime( char* buffer , size_t max , const char* tfmt , const char* afmt )
{
  const size_t hmax = 80 ;
  char hbuf[hmax] ;
  int hrc = gethostname( hbuf , hmax );

  const size_t tmax = 80 ;
  char tbuf[tmax] ;
  int trc = private_getftime( tbuf , tmax , tfmt );
   
  snprintf( buffer, max , afmt , hbuf , tbuf );
  return trc + hrc ;
}


int main(int argc, char** argv)
{
    const size_t max = 80 ;
    char buf[max] ;
    private_gethostftime( buf , max , "%c" , "%s %s" ) ;
    printf( buf );

    int rc ;
    rc = private_init(); if(rc != EXIT_SUCCESS) exit(rc) ;
    int a ; for ( a = 1; a < argc; a++ ) printf("%s\n", private_lookup(argv[a])) ;
    rc = private_cleanup(); if(rc != EXIT_SUCCESS) exit(rc) ;
    return EXIT_SUCCESS;
}

