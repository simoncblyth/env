#ifndef JS_H
#define JS_H

#include <string>
#include <map>
typedef std::map<std::string,std::string> Map_t ;

struct cJSON ;

class JS {
public:
   static JS* Load(const char* path);

   static const char* itype ;
   static const char* ftype ;
   static const char* stype ;
   static const char* btype ;

public:
   JS(const char* text);
   virtual ~JS();

   void Print(const char* msg="JS::Print");
   void PrintToFile(const char* path);

   void SetVerbosity(int verbosity);
   int GetVerbosity();

   // mode:0 
   //    initial traversal collecting all strings
   //    matching the sentinel into the selection map
   //
   // mode:1 
   //    subsequent traversal operates 
   //    based on the types found
   //
   void SetMode(int mode);
   int GetMode();

public:
   void DumpMap(Map_t& map, const char* msg);
   Map_t CreateRowMap();
   Map_t CreateTypeMap();
   Map_t CreateMap(char form);

public:
   // JSON operations
   const char* TypeName( char type);
   char Type( int type);
   void Traverse(const char* wanted);
   void Recurse(cJSON* item, const char* prefix, const char* wanted);
   void Visit(cJSON *item, const char* prefix, const char* wanted);
   void AddKV(cJSON* obj, const char* key, const char* val );
   void AddMap(const char* name, Map_t& map);
   void FindTypes(const char* sentinel="COLUMNS");
   void DumpTypes();
   char LookupType(const char* path);
   void DumpItem( cJSON* item, const char* prefix  );

public:
   // Map_t manipulations, eg for selecting parts of the JSON tree
   void AddMapKV( const char* key, const char* val );
   void ClearMap();
   void PrintMap(const char* msg="JS::PrintMap");
   Map_t& GetMap();


private: 
   cJSON* m_root ; 
   Map_t m_map ;  
   Map_t m_type ;  
   int m_verbosity ;
   int m_mode ;

};


#endif 

