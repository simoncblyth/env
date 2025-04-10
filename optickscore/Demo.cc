#include "Demo.hh"
#include "GLMFormat.hpp"

const char* Demo::PREFIX = "demo" ;

const char* Demo::A = "a" ;
const char* Demo::B = "b" ;
const char* Demo::C = "c" ;


const char* Demo::getPrefix()
{    
   return PREFIX ; 
}






Demo::Demo() 
  :
   m_a(0.f),
   m_b(0.f),
   m_c(0.f)
{
}

float Demo::getA(){ return m_a ; }
float Demo::getB(){ return m_b ; }
float Demo::getC(){ return m_c ; }

void Demo::setA(float a){ m_a = a ; }
void Demo::setB(float b){ m_b = b ; }
void Demo::setC(float c){ m_c = c ; }

std::vector<std::string> Demo::getTags()
{
    std::vector<std::string> tags ;
    tags.push_back(A);
    tags.push_back(B);
    //tags.push_back(C);
    return tags ; 
}







std::string Demo::get(const char* name)
{
    float v(0.f) ; 
   
    if(     strcmp(name,A)==0)     v = getA();
    else if(strcmp(name,B)== 0 )   v = getB();
    else if(strcmp(name,C)== 0 )   v = getC();
    else
         printf("Demo::get bad name %s\n", name);

    return gformat(v);
}

void Demo::set(const char* name, std::string& s)
{
    float v = gfloat_(s);

    if(     strcmp(name,A)==0)    setA(v);
    else if(strcmp(name,B)== 0 )  setB(v);
    else if(strcmp(name,C)== 0 )  setC(v);
    else
         printf("Demo::set bad name %s\n", name);
}



void Demo::configureS(const char* , std::vector<std::string> )
{
}
void Demo::configureF(const char*, std::vector<float>  )
{
}
void Demo::configureI(const char* , std::vector<int>  )
{
}




