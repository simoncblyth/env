/*

   cc cJSON.c testcjson.c   -o $LOCAL_BASE/env/bin/testcjson && testcjson $PWD/out.js

*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "cJSON.h"


char* slurp( const char* filename )
{
    FILE *f=fopen(filename,"rb");
    fseek(f,0,SEEK_END);
    long len=ftell(f);
    fseek(f,0,SEEK_SET);

    char *data=(char*)malloc(len+1);
    fread(data,1,len,f);
    fclose(f);
    data[len] = '\0' ;

    return data;
}


cJSON* parse(const char* filename, int pretty)
{
    char* text = slurp(filename);
    cJSON* root = cJSON_Parse(text);
    free(text);

    if(pretty)
    {
        char *out = cJSON_Print(root);
        printf("%s\n",out);
        free(out);
    }
    return root ;
}


void visit(cJSON *item, const char* prefix)
{
    switch( item->type ){
        case cJSON_Object:break;
        case cJSON_Array:break;
        case cJSON_Number:printf("N: %s %d %f \n", prefix, item->valueint, item->valuedouble); break ;
        case cJSON_String:printf("S: %s %s \n", prefix, item->valuestring); break ;
        default:printf("?: %s type %d \n", prefix, item->type );
    }
}


void recurse(cJSON *item, const char* prefix )
{
    while (item)
    {
        char* newprefix = NULL ; 
        const char* key = item->string ; 
        if(key)
        {
           newprefix = malloc(strlen(prefix)+strlen(key)+2);
           sprintf(newprefix,"%s/%s",prefix,key);
        }
        else
        {
           newprefix = malloc(strlen(prefix)+1);
           sprintf(newprefix,"%s",prefix);
        }
        visit(item, newprefix);
        if(item->child) recurse(item->child, newprefix);
        item=item->next;
        free(newprefix);
    }
}


int main(int argc, char** argv)
{
    assert(argc > 1);
    char* filename = argv[1] ; 
    cJSON* root = parse(filename,0);

    recurse(root, "");

    cJSON_Delete(root);
    return 0 ;
}

