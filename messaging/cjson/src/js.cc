/*
   See cjs- for building 
  
  pfx=$(cjs-prefix) && clang js.cc jstest.cc -lstdc++ -I$pfx/include -L$pfx/lib -lcJSON -Wl,-rpath,$pfx/lib -o $LOCAL_BASE/env/bin/js && js out.js

*/

#include "cJSON/js.hh"

#include <cassert>
#include <cstdlib>
#include <stddef.h>
#include <stdio.h>
#include <sstream>
#include <vector>
#include <string>

#include "cJSON/cJSON.h"

// functions defined at tail
char* slurp( const char* filename );
void split( std::vector<std::string>& elem, const char* line, char delim );


// statics

const char* JS::INTEGER_TYPE = "integer" ;
const char* JS::FLOAT_TYPE = "real" ;
const char* JS::STRING_TYPE = "text" ;
const char* JS::BLOB_TYPE = "blob" ;
const char* JS::SENTINEL = "COLUMNS" ;

JS* JS::Load(const char* path)
{
    char* text = slurp(path);
    if(!text) return NULL ;
    JS* js = new JS(text);
    free(text);
    return js ; 
}

// lifecycle

JS::JS(const char* text) : m_root(NULL), m_verbosity(0), m_mode(0)
{
    assert(text);
    m_root = cJSON_Parse(text);
    Analyse();
}

JS::~JS()
{
    cJSON_Delete(m_root);
}

//  primary operations

void JS::AddMap(const char* name, Map_t& map)
{
    cJSON* obj = cJSON_CreateObject();
    for(Map_t::iterator it=map.begin(); it != map.end() ; ++it ) AddKV(obj, it->first.c_str(),it->second.c_str()) ;
    cJSON_AddItemToObject(m_root,name,obj);
    Analyse();
}



Map_t JS::CreateSubMap(const char* wanted)
{
    return CreateMap('r', NULL, wanted);
}

Map_t JS::CreateRowMap(const char* columns)
{
    return CreateMap('r', columns, NULL);
}

Map_t JS::CreateTypeMap(const char* columns)
{
    return CreateMap('t', columns, NULL);
}

Map_t JS::CreateMap(char form, const char* columns, const char* wanted)
{
   std::vector<std::string> cols ;
   if(columns) split(cols, columns, ','); 

    Map_t xmap ; 
    for(Map_t::iterator it=m_map.begin() ; it != m_map.end() ; it++ )
    {
        const char* key = it->first.c_str();
        const char* val = it->second.c_str();
        char look = LookupType(key);
        const char* type = TypeName(look);
        const char* name = strrchr(key, '/') + 1;  // keys are full paths within the js tree, this plucks just the basename

        bool select = true ;
        if(!cols.empty())
        {
            select = std::find(cols.begin(), cols.end(), name) != cols.end() ;
        } 

        if(wanted)
        {  
            if(wanted[0] == '/') // absolute: match all "wanted" from start 
            {  
                select = strncmp( key, wanted, strlen(wanted)) == 0  ;
            }
            else   // relative: match "wanted" from end
            {
                select = strncmp( key + strlen(key) - strlen(wanted), wanted, strlen(wanted)) == 0  ;
            }
        }


        if(!select)
        {
           continue;
        }


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

    if(xmap.size() != m_map.size() && columns == NULL && wanted == NULL)
    {
        printf("JS::CreateMap [%c] xmap %zu m_map %zu map size mismatch \n", form, xmap.size(),m_map.size() );
        DumpMap(xmap,  "xmap"); 
        DumpMap(m_map, "m_map"); 
    }
    return xmap ;
} 







// secondary operations

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

void JS::Print(const char* msg)
{
    printf("%s\n", msg);
    if(!m_root) return ;

    char *out = cJSON_Print(m_root);
    printf("%s\n",out);
    free(out);
}

std::string JS::AsString(bool pretty)
{
    char *out = pretty ? cJSON_Print(m_root) : cJSON_PrintUnformatted(m_root) ;
    std::string str(out);
    free(out);
    return str; 
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

void JS::Traverse(const char* wanted)
{
   // used by interative JSON dumper : "which js"
    ClearMap();
    Recurse(m_root, "", wanted );
    PrintMap("JS::Traverse");
}

void JS::PrintMap(const char* msg) const
{
    printf("%s\n", msg);
    for(Map_t::const_iterator it=m_map.begin() ; it != m_map.end() ; it++ )
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


// tertiary operations


void JS::Demo()
{
    Map_t tmap = CreateTypeMap();
    Map_t rmap = CreateRowMap();

    DumpMap(tmap, "typemap");
    DumpMap(rmap, "rowmap");
}

void JS::SetVerbosity(int verbosity)
{
    m_verbosity = verbosity ;
}
int JS::GetVerbosity()
{
    return m_verbosity;
}



// high level internals :  convert from JSON tree into maps

void JS::Analyse()
{
   /*
      mode:0 
         initial traversal collecting all strings
         matching the sentinel into the selection map
   
      mode:1 
         subsequent traversal operates 
         based on the types found
   
   */

    SetMode(0);
    ClearMap();
    Recurse(m_root, "", SENTINEL );   // m_root -> m_map (recurse selects sentinel paths only)

    ParseSentinels();                 // m_map -> m_type 
    if(m_verbosity > 1) DumpMap(m_type, "m_type");

    SetMode(1);
    ClearMap();
    Recurse(m_root, "", "" );         // m_root -> m_map  (full tree recurse, selecting items with defined types)
}

void JS::ParseSentinels()
{
    /*
        Sentinel m_map (key,val) entries like:: 

             "/parameters/COLUMNS" : "name:s,nphotons:i,nwork:i,nsmall:i,npass:i,nabort:i,nlaunch:i,tottime:f,maxtime:f,mintime:f"

        Are transformed into m_type entries

             "/parameters/name"      : "s"
             "/parameters/nphotons"  : "i"
             ...


    */
    for(Map_t::iterator it=m_map.begin() ; it != m_map.end() ; it++ )
    {
        const char* key = it->first.c_str();
        const char* val = it->second.c_str();

        size_t size = strlen(key) - strlen(SENTINEL) ; 
        std::string pfx(key, size );   

        std::vector<std::string> columns; 
        split(columns, val, ',' );     

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
}

void JS::SetMode(int mode)
{
    m_mode = mode ;
}
int JS::GetMode()
{
    return m_mode;
}


// m_map manipulations

void JS::ClearMap()
{
    m_map.clear();
}

void JS::AddMapKV( const char* key, const char* val )
{
    std::string k(key);
    std::string v(val);
    m_map[k] = v ;
}

Map_t JS::GetMap(const char* wanted)
{
    return wanted ? CreateSubMap(wanted) : m_map ;
}


// navigating JS tree


void JS::Visit(cJSON *item, const char* prefix, const char* wanted )
{
    if(m_verbosity > 1) DumpItem(item, prefix);

    if(m_mode == 0 )  // just plucking sentinel strings ie COLUMNS with sqlite type info
    {
        if(item->type == cJSON_String ) AddMapKV(prefix, item->valuestring );
        return;
    }


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


// adding to JS tree
void JS::SetKV(const char* name, const char* key, const char* val )
{
     cJSON* obj = cJSON_GetObjectItem(m_root, name);
     if(!obj)
     {
         obj = cJSON_CreateObject();
         cJSON_AddItemToObject(m_root,name,obj);
         //printf("JS::SetKV create top level object named %s \n", name);
     }
     AddKV(obj, key, val);
     Analyse();
}


/*
std::string JS::Get(const char* name, const char* key)
{
     std::string ret ;
     cJSON* obj  = cJSON_GetObjectItem(m_root, name);
     if(!obj) return ret ;

     cJSON* item = cJSON_GetObjectItem(obj, name);
     if(!item) return ret ;

     ret.assign(item->valuestring);
     return ret ;
}
*/



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

// lookups

char JS::LookupType(const char* path) const
{
    char ret = ' ' ;

    std::string key(path);
    Map_t::const_iterator end = m_type.end();
    Map_t::const_iterator ptr = m_type.find(key);

    if(ptr != end) ret = *ptr->second.c_str();
    return ret ;    
}

const char* JS::TypeName( char type) const
{
    const char* r = NULL ;
    switch( type )
    {
        case 'i':r = INTEGER_TYPE ;break;
        case 'f':r = FLOAT_TYPE ;break;
        case 's':r = STRING_TYPE ;break;
         default:r = BLOB_TYPE ;break; 
    }
    return r ;    
}

char JS::Type( int type) const
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


// debug internals

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





// tail functions

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


