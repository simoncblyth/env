/*
  pfx=$(cjs-prefix) && clang js.cc jstest.cc -lstdc++ -I$pfx/include -L$pfx/lib -lcJSON -Wl,-rpath,$pfx/lib -o $LOCAL_BASE/env/bin/js && js out.js

*/

#include "cJSON/js.hh"

#include <cassert>
#include <cstdlib>
#include <stddef.h>
#include <stdio.h>
#include <sstream>
#include <vector>

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


void split( std::vector<std::string>& elem, const char* line, char delim )
{
    if(line == NULL) return ;
    std::istringstream f(line);
    std::string s;
    while (getline(f, s, delim)) elem.push_back(s);
}


const char* JS::itype = "integer" ;
const char* JS::ftype = "real" ;
const char* JS::stype = "text" ;
const char* JS::btype = "blob" ;


JS::JS(const char* text) : m_root(NULL), m_verbosity(3), m_mode(0)
{
    if(text){
        m_root = cJSON_Parse(text);
        FindTypes();
    }
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


void JS::SetVerbosity(int verbosity)
{
    m_verbosity = verbosity ;
}

int JS::GetVerbosity()
{
    return m_verbosity;
}

void JS::SetMode(int mode)
{
    m_mode = mode ;
}

int JS::GetMode()
{
    return m_mode;
}


char JS::LookupType(const char* path)
{
    std::string key(path);
    char ret = ' ' ;
    if(m_type.find(key) != m_type.end()) ret = *m_type[key].c_str();
    return ret ;    
}

void JS::FindTypes(const char* sentinel)
{
    SetMode(0);
    ClearMap();
    Recurse(m_root, "", sentinel );   // recursion collects into m_map
    SetMode(1);

    for(Map_t::iterator it=m_map.begin() ; it != m_map.end() ; it++ )
    {
        const char* key = it->first.c_str();
        const char* val = it->second.c_str();

        size_t size = strlen(key) - strlen(sentinel) ; 
        std::string pfx(key, size );   //  eg "/parameters/" from "/parameters/COLUMNS"

        std::vector<std::string> columns; 
        split(columns, val, ',' );        // val example "name:s,nphotons:i,nwork:i,nsmall:i,npass:i,nabort:i,nlaunch:i,tottime:f,maxtime:f,mintime:f"

        for(size_t c=0 ; c < columns.size() ; ++c)
        {
             std::vector<std::string> pair ; 
             split(pair, columns[c].c_str(), ':');
             assert(pair.size() == 2);

             std::string path(pfx);
             path += pair[0] ;
             m_type[path] = pair[1] ;
        }          

        if(m_verbosity > 2) printf(" %40s : %20s : %s  \n", key, val, pfx.c_str() );
    }

    if(m_verbosity > 1) DumpTypes();
}

void JS::DumpTypes()
{
    for(Map_t::iterator it=m_type.begin() ; it != m_type.end() ; it++ )
    {
        const char* key = it->first.c_str();
        const char* val = it->second.c_str();
        printf(" [%s] %s \n", val, key ); 
    } 
}

void JS::Traverse(const char* wanted)
{
    ClearMap();
    Recurse(m_root, "", wanted );
    PrintMap("JS::Traverse");

    Map_t tmap = CreateTypeMap();
    Map_t rmap = CreateRowMap();

    DumpMap(tmap, "typemap");
    DumpMap(rmap, "rowmap");
}

const char* JS::TypeName( char type)
{
    const char* r = NULL ;
    switch( type )
    {
        case 'i':r = itype ;break;
        case 'f':r = ftype ;break;
        case 's':r = stype ;break;
         default:r = btype ;break; 
    }
    return r ;    
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



void JS::DumpItem( cJSON* item, const char* prefix  )
{
    const char* key = item->string ? item->string : "~" ;  // anonymous nodes like root have empty key 
    char lookup = LookupType(prefix);
    char type = Type(item->type);
    printf("%c [%c] %40s %20s  ",type, lookup, prefix, key );
    switch( type ){
        case 'O':break;
        case 'A':break;
        case 'F':printf(" %d %f ", item->valueint, item->valuedouble); break ;
        case 'T':printf(" %s ", item->valuestring); break ;
        case 'B':printf(" %d ", item->valueint );break;
        case 'N':printf(" nul ");break;
        default:printf("?????: item->type %d ", item->type );
    }
    printf("\n");
}


void JS::Visit(cJSON *item, const char* prefix, const char* wanted )
{
    if(m_verbosity > 1) DumpItem(item, prefix);
    if(m_mode == 0 )
    {
        if(item->type == cJSON_String ) AddMapKV(prefix, item->valuestring );
        return;
    }
    //const char* key = item->string ? item->string : "~" ;  // anonymous nodes like root have empty key 
    char look = LookupType(prefix); // type char obtained from sentinel fields
    size_t size = 256 ;
    char* value = new char[size];
    bool skip = false ; 
    switch(look)
    {
        case 'i':
                 snprintf(value, size,  "%d", item->valueint );
                 break;
        case 'f':
                 snprintf(value, size,  "%f", item->valuedouble );
                 break;
        case 's':
                 snprintf(value, size,  "%s", item->valuestring );
                 break;
        default:
                 skip = true ;
                 break;
              
    }

    if(!skip) AddMapKV(prefix, value);
}



void JS::AddMapKV( const char* key, const char* val )
{
    std::string k(key);
    std::string v(val);
    m_map[k] = v ;
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

        bool match = false ;
        if(wanted[0] == '/') // absolute: match all "wanted" from start 
        {  
            match = strncmp( newprefix, wanted, strlen(wanted)) == 0  ;
        }
        else   // relative: match "wanted" from end
        {
            match = strncmp( newprefix + strlen(newprefix) - strlen(wanted), wanted, strlen(wanted)) == 0  ;
        }
        if(match) Visit(item, newprefix, wanted );
        

        if(item->child) Recurse(item->child, newprefix, wanted );
        item=item->next;
        free(newprefix);
    }
}

void JS::ClearMap()
{
    m_map.clear();
}

void JS::PrintMap(const char* msg)
{
    printf("%s\n", msg);
    for(Map_t::iterator it=m_map.begin() ; it != m_map.end() ; it++ )
    {
        const char* key = it->first.c_str();
        const char* val = it->second.c_str();
        char look = LookupType(key);
        const char* name = strrchr(key, '/') + 1;
        const char* type = TypeName(look);

        if(!name) name="" ;
        printf(" [%c]%10s %40s : %20s : %s \n", look,type, key, name, val );
    }
}
void JS::DumpMap(Map_t& map, const char* msg)
{
    printf("JS::DumpMap %s \n", msg);
    for(Map_t::iterator it=map.begin() ; it != map.end() ; it++ )
    {
        const char* key = it->first.c_str();
        const char* val = it->second.c_str();
        printf(" %20s : %s \n", key, val );
    }
}
Map_t JS::CreateRowMap()
{
    return CreateMap('r');
}
Map_t JS::CreateTypeMap()
{
    return CreateMap('t');
}
Map_t JS::CreateMap(char form)
{
    Map_t xmap ; 
    for(Map_t::iterator it=m_map.begin() ; it != m_map.end() ; it++ )
    {
        const char* key = it->first.c_str();
        const char* val = it->second.c_str();
        char look = LookupType(key);
        const char* type = TypeName(look);
        const char* name = strrchr(key, '/') + 1;

        switch(form)
        {
           case 't':
                    xmap[std::string(name)] =  std::string(type);
                    break;
           case 'r':
                    xmap[std::string(name)] =  std::string(val);
                    break;
        }
    }   
    //assert( xmap.size() == m_map.size() ); // possibly non-unique name 

    if(xmap.size() != m_map.size())
    {
        printf("JS::CreateMap [%c] xmap %zu m_map %zu map size mismatch \n", form, xmap.size(),m_map.size() );
        DumpMap(xmap,  "xmap"); 
        DumpMap(m_map, "m_map"); 
    }
    return xmap ;
} 



Map_t& JS::GetMap()
{
    return m_map ; 
}


void JS::AddKV(cJSON* obj, const char* key, const char* val )
{
   // endptr pointing to null terminator means converted while string 
   {
      char* endptr;
      long int lval = strtol(val, &endptr, 10); 
      if(!*endptr)  
      {
          cJSON_AddNumberToObject(obj, key, lval );
          return ; 
      }
   }
   {
      char* endptr;
      double dval = strtod(val, &endptr); 
      if(!*endptr)  
      {
          cJSON_AddNumberToObject(obj, key, dval );
          return ; 
      }
   }
   cJSON_AddStringToObject(obj, key, val );
}


void JS::AddMap(const char* name, Map_t& map)
{
    cJSON* obj = cJSON_CreateObject();
    for(Map_t::iterator it=map.begin(); it != map.end() ; ++it ) AddKV(obj, it->first.c_str(),it->second.c_str()) ;
    cJSON_AddItemToObject(m_root,name,obj);

    FindTypes();
}


void JS::Print(const char* msg)
{
    printf("%s\n", msg);
    if(!m_root) return ;

    char *out = cJSON_Print(m_root);
    printf("%s\n",out);
    free(out);
}



void JS::PrintToFile(const char* path)
{
    char *out = cJSON_Print(m_root);
    FILE* fp=fopen(path,"w");
    if(!fp){
        fprintf(stderr, "JS::PrintToFile failed to open for writing:  %s \n", path);
        return ;
    }
    printf("JS::PrintToFile %s\n",path);
    fprintf(fp,"%s\n", out);
    fclose(fp);
    free(out);
}



