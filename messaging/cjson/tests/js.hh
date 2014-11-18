#ifndef JS_H
#define JS_H

#include <string>
#include <map>
typedef std::map<std::string,std::string> Map_t ;

struct cJSON ;

class JS {
public:
   static JS* Load(const char* path);

public:
   JS(const char* text);
   virtual ~JS();

   char Type( int type);
   void Print(const char* msg="JS::Print");
   void Traverse(const char* wanted);
   void Recurse(cJSON* item, const char* prefix, const char* wanted);
   void Visit(cJSON *item, const char* prefix, const char* wanted);

   void PopulateMap(const char* prefix=NULL);
   void DumpMap();

   //Map_t& GetMap();
   //Map_t  GetMapCopy();


private:
   cJSON* m_root ; 
   Map_t m_map ;  

};


#endif 

