/*
   cc   -I$(jsmn-dir) -L$(jsmn-dir) -ljsmn jsmntest.c -o $LOCAL_BASE/env/bin/jsmntest && jsmntest
*/

#include <string.h>
#include <stdio.h>
#include "jsmn.h"




int parse_json( const char* js, jsmntok_t* tokens, unsigned int maxtokens  )
{
    jsmn_parser parser;
    jsmn_init(&parser);
    int rc = jsmn_parse(&parser, js, strlen(js), tokens, maxtokens);
    if(rc < 0)
    {
       switch (rc) {
          case JSMN_ERROR_INVAL:printf("JSMN_ERROR_INVAL\n"); break ;     
          case JSMN_ERROR_NOMEM:printf("JSMN_ERROR_NOMEM\n"); break ;     
          case JSMN_ERROR_PART:printf("JSMN_ERROR_PART\n"); break ;     
          default:printf("JSMN unexpected RC %d\n", rc);
       }
    } 
    return rc ;
}





int main()
{

    const char* js = "{ \"name\" : \"Jack\", \"age\" : 27 }" ;
    printf("%s\n",js);

    const unsigned int ntok = 256 ;
    jsmntok_t tokens[ntok];

    int rc = parse_json( js, tokens, ntok );
    if(rc < 0) return rc ;
   
    printf("rc %d\n", rc ); 

    char tmp[100]; 

    for(int i=0 ; i < rc ; ++i )
    {
        jsmntok_t tok = tokens[i];
        printf("i %d start %d end %d size %d \n", i, tok.start, tok.end, tok.size ); 

        int len = tok.end - tok.start ;
        if( len < 100 ){
           strncpy (tmp, js + tok.start, len );
           tmp[len] = '\0' ;
           printf("i %d start %d end %d size %d tok %s \n", i, tok.start, tok.end, tok.size, tmp ); 
        }        

    }  


    return 0 ; 
}

