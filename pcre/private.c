/*

    gcc -Wall private.c -I/opt/local/include -L/opt/local/lib  -lpcre -o private

*/


#include <stdio.h>
#include <stdlib.h>
#include <pcre.h>

#define OVECCOUNT 30    /* should be a multiple of 3 */

int main(void)
{
 
    const char *patn = "^local (?P<name>.*)=(?P<value>.*)" ;
    const char *errstr ;
    int erroff;
    pcre *re;
    if (!(re = pcre_compile(patn, 0, &errstr, &erroff, 0))) {
       fprintf(stderr, "%s: %s\n", patn, errstr );
       return EXIT_FAILURE;
    }
       
    const char *name = "ENV_PRIVATE_PATH";
    char *path = getenv(name);
    FILE *file = fopen(path, "r");
    char line[512];  

	unsigned char *name_table;
	int erroffset;
	int namecount;
	int name_entry_size;
	int ovector[OVECCOUNT];
	int rc, i ;

    if ( file ){
        while ( fgets(line, sizeof line, file) ){
             
           size_t line_len = strcspn(line, "\n");
 		   rc = pcre_exec( re,  NULL, line, line_len, 0, 0, ovector,  OVECCOUNT); 
           printf("%d %s\n", rc, line ); 

		   if (rc <= 0)
		   {
		         switch(rc){
			  		  case 0:printf("Too many matches for ovector \n"); break;
		              case PCRE_ERROR_NOMATCH: printf("No match\n"); break;
		                              default: printf("Matching error %d\n", rc); break;
		         } 
		   }
		   else 
		   {
		         printf("\nMatch succeeded at offset %d with rc %d \n", ovector[0], rc );	
			     for (i = 0; i < rc; i++) {
			         char *substring_start = line + ovector[2*i];
			         int substring_length = ovector[2*i+1] - ovector[2*i];
			         printf("%2d: %.*s\n", i, substring_length, substring_start);
			     }
			
				(void)pcre_fullinfo(re, NULL, PCRE_INFO_NAMECOUNT, &namecount);  // number of named substrings
				if (namecount <= 0) printf("No named substrings\n"); else
				{
				     unsigned char *tabptr;
				     printf("Named substrings\n");

				     /* Before we can access the substrings, we must extract the table for
				       translating names to numbers, and the size of each entry in the table. */

				     (void)pcre_fullinfo(re, NULL, PCRE_INFO_NAMETABLE, &name_table);          
				     (void)pcre_fullinfo(re, NULL, PCRE_INFO_NAMEENTRYSIZE, &name_entry_size);    
        
				     tabptr = name_table;
				     for (i = 0; i < namecount; i++)
				     {
				         int n = (tabptr[0] << 8) | tabptr[1];
				         printf("(%d) %*s: %.*s\n", n, name_entry_size - 3, tabptr + 2, ovector[2*n+1] - ovector[2*n], line + ovector[2*n]);
				         tabptr += name_entry_size;
				     }
				  }	
			
		   }
		
       }
  }

  
  //pcre_free(re); 
  return EXIT_SUCCESS;

}

