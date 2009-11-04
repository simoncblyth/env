// 
//  based on http://wiki.hypexr.org/wikka.php?wakka=RegexExamples
//
// alternative would be to use ini file parsing 
// http://c.snippets.org/snip_lister.php?fname=ini.c
//
//
// http://www.ddj.com/web-development/184402021
//
//
//
//  perl -n -e 'm/^local (\w*)=(\S*)/ && printf "$1 $2 \n" ; ' ~/.bash_private
//
//


#include <stdio.h>
#include <stdlib.h>

#include <pcre.h>

char enter_reverse_mode[] = "\33[7m";
char exit_reverse_mode[] = "\33[0m";

int main(void)
{
    const char *name = "ENV_PRIVATE_PATH";
    char *path = getenv(name);

    char line[512];
    FILE *file = fopen(path, "r");

    const char *patn = "^local " ;
    const char *errstr ;
    int erroff;
    pcre *re;

    if (!(re = pcre_compile(patn, 0, &errstr, &erroff, 0))) {
       fprintf(stderr, "%s: %s\n", patn, errstr );
        return EXIT_FAILURE;
    }
         
    if ( file ){
        while ( fgets(line, sizeof line, file) ){
           //printf("%s\n", line );   

           size_t len = strcspn(line, "\n");
           int matches[2];
           int offset = 0;
           int flags = 0;
           line[len] = '\0';
           // offset specifies where to start matching in the string
           while (0 < pcre_exec(re, 0, line, len, offset, flags, matches, 2)) {   
                printf("%.*s%s%.*s%s", matches[0] - offset, line + offset, enter_reverse_mode, matches[1] - matches[0], line + matches[0], exit_reverse_mode);
                offset = matches[1];
                flags |= PCRE_NOTBOL;   // PCRE_NOTBOL : Subject is not the beginning of a line 
           }
           printf("%s\n", line + offset);
       
        }
  }
  return EXIT_SUCCESS;

}




