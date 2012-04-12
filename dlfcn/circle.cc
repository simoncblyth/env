
// http://www.linuxjournal.com/article/3687

#include <iostream> 
#include <map>
#include <string>

#include "circle.hh"

void circle::draw(){
   cout << "\n";
   cout << "      ****\n";
   cout << "    *      *\n";
   cout << "   *        *\n";
   cout << "   *        *\n";
   cout << "   *        *\n";
   cout << "    *      *\n";
   cout << "      ****\n";
   cout << "\n";
}

extern "C" {

shape* maker(){ return new circle; }

class proxy { 
public:
   proxy(){ factory["circle"] = maker; } 
};
proxy p;  // instanciated on dlopen(lib, RTLD_NOW)  which then registers in global factory map


}

