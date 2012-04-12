
// http://www.linuxjournal.com/article/3687

#include <iostream> 
#include <map>
#include <string>
#include "square.hh"


void square::draw(){
   cout << " SQUARE " << endl ;
   cout << " SQUARE " << endl ;
   cout << " SQUARE " << endl ;
   cout << " SQUARE " << endl ;
   cout << " SQUARE " << endl ;
   cout << " SQUARE " << endl ;
}
extern "C" {

shape* maker(){ return new square; }
class proxy { 
public:
   proxy(){ factory["square"] = maker; } 
};

// instanciated on dlopen(lib, RTLD_NOW)  
// which then registers in global factory map
proxy p;

}


