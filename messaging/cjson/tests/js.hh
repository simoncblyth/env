#ifndef JS_H
#define JS_H

struct cJSON ;

class JS {
public:
   static JS* Load(const char* path);

public:
   JS(const char* text);
   virtual ~JS();

   void Print(const char* msg="JS::Print");

private:
   cJSON* m_root ; 

};


#endif 

