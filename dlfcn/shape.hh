#ifndef __SHAPE_H
#define __SHAPE_H
#include <map>
#include <string>
using namespace std ;

class shape {
public:
   virtual void draw()=0;
};

typedef shape* maker_ptr();   // maker_ptr is a function that takes void and returns shape*

// global registry of maker_ptr 
extern map<string, maker_ptr*, less<string> > factory;


#endif 
