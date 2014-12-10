// cc testrealloc.cc  -lstdc++ -o testrealloc && ./testrealloc && rm testrealloc
// cc testrealloc.cc -g -lstdc++ -o testrealloc && lldb testrealloc 
//
//  http://forum.codecall.net/topic/51010-dynamic-arrays-using-malloc-and-realloc/

#include "stdint.h"
#include "stdlib.h"
#include "stdio.h"
#include "assert.h"


class A {
  public:
     A(size_t capacity, float m_factor) ;
     virtual ~A();
   
  public:
     void Print(const char* msg);
     void Append(int val);
     void Dump(const char* msg);
 
  private:
     void Extend(size_t capacity);
     int* GetNextPointer();

  private:
     int*   m_data ; 
     size_t m_capacity ; 
     float  m_factor ; 
     size_t m_size ; 
     size_t m_itemsize ; 

};


void A::Print(const char* msg)
{
    printf("%s capacity %zu size %zu itemsize %zu \n", msg, m_capacity, m_size, m_itemsize ); 
}


A::A(size_t capacity, float factor) : m_data(NULL), m_capacity(capacity), m_factor(factor), m_size(0), m_itemsize(sizeof(int)) 
{
    m_data = (int*)malloc( m_capacity*m_itemsize ) ;
}

void A::Extend(size_t capacity)
{
   int* tmp = (int*)realloc(m_data, capacity*m_itemsize  );
   if(tmp) 
   {
       //printf("A::Extend factor %10.2f increase capacity from %zu ->  %zu \n", m_factor, m_capacity, capacity );
       m_data = tmp;
       m_capacity = capacity ; 
   }
   else 
   {
       printf("A::Extend FAILURE factor %10.2f increase capacity from %zu ->  %zu \n", m_factor, m_capacity, capacity );
   }
} 


A::~A()
{
    free(m_data);
}



int* A::GetNextPointer()
{
    if(m_size == m_capacity)
    {
        Extend(m_capacity*m_factor);
    }

    if (m_size < m_capacity )
    {
        int* p = m_data + m_size ;   // no m_itemsize as m_data is int*
        m_size++ ;
        return p ;
    }
    return NULL ;
}


void A::Append(int val)
{ 
    int* p = GetNextPointer();
    if(!p){
        printf("A::Append failed to GetNextPointer \n");
        return ;
    }

    *p = val ;
}

void A::Dump(const char* msg )
{
    printf("A::Dump %s\n", msg );

    for(size_t i = 0 ; i < m_size ; i++ )
    {
        size_t j = m_data[i];
        //printf(" %zu : %zu \n", i, j );
        assert(i == j);
    }

}



int main()
{
    A* a = new A(10, 1.5);
    for(size_t i=0 ; i < 1000 ; i++) a->Append(i);
    a->Dump("init");
}



