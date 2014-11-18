/*
  pfx=$(cjs-prefix) && clang js.cc jstest.cc -lstdc++ -I$pfx/include -L$pfx/lib -lcJSON -Wl,-rpath,$pfx/lib -o $LOCAL_BASE/env/bin/js && js out.js

*/

#include "js.hh"

#include <cstdlib>
#include <stddef.h>
#include <stdio.h>

#include "cJSON/cJSON.h"


char* slurp( const char* filename )
{
    FILE *f=fopen(filename,"rb");
    if(!f)
    {
        fprintf(stderr, "slurp: failed to open %s \n", filename);
        return 0 ;
    }
    fseek(f,0,SEEK_END);
    long len=ftell(f);
    fseek(f,0,SEEK_SET);

    char *data=(char*)malloc(len+1);
    fread(data,1,len,f);
    fclose(f);
    data[len] = '\0' ;

    return data;
}


JS::JS(const char* text) : m_root(NULL)
{
    if(text) m_root = cJSON_Parse(text);
}

JS::~JS()
{
    if(m_root) cJSON_Delete(m_root);
}

JS* JS::Load(const char* path)
{
    char* text = slurp(path);
    if(!text) return NULL ;
    JS* js = new JS(text);
    free(text);
    return js ; 
}





void JS::Traverse(const char* wanted)
{
    Recurse(m_root, "", wanted );
}


char JS::Type( int type)
{
    char rc = '~' ;
    switch( type ){
        case cJSON_False:
        case cJSON_True  : rc='B';break;
        case cJSON_NULL  : rc='N';break; 
        case cJSON_Object: rc='O';break;
        case cJSON_Array : rc='A';break;
        case cJSON_Number: rc='F';break;
        case cJSON_String: rc='T';break;
        default:rc='?';break;
    }
    return rc ; 
}


void JS::Visit(cJSON *item, const char* prefix, const char* wanted )
{
    char type = Type(item->type);
    printf("%c %40s ",type, prefix );

    switch( type ){
        case 'O':break;
        case 'A':break;
        case 'F':printf(" %d %f ", item->valueint, item->valuedouble); break ;
        case 'T':printf(" %s ", item->valuestring); break ;
        case 'B':printf(" %d ", item->valueint );break;
        default:printf("?????: item->type %d ", item->type );
    }
    printf("\n");

}

void JS::Recurse(cJSON *item, const char* prefix, const char* wanted )
{
    while (item)
    {
        char* newprefix = NULL ; 
        const char* key = item->string ; 
        if(key)
        {
           newprefix = (char*)malloc(strlen(prefix)+strlen(key)+2);
           sprintf(newprefix,"%s/%s",prefix,key);
        }
        else
        {
           newprefix = (char*)malloc(strlen(prefix)+1);
           sprintf(newprefix,"%s",prefix);
        }

        if(strncmp( newprefix, wanted, strlen(wanted)) == 0)
        {
            Visit(item, newprefix, wanted );
        }

        if(item->child) Recurse(item->child, newprefix, wanted );
        item=item->next;
        free(newprefix);
    }
}

void JS::PopulateMap(const char* starting)
{
}

void JS::DumpMap()
{
}

void JS::Print(const char* msg)
{
    printf("%s\n", msg);
    if(!m_root) return ;

    char *out = cJSON_Print(m_root);
    printf("%s\n",out);
    free(out);
}


