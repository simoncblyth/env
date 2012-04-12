// listing3
#include <iostream>
#include "square.hh"
void square::draw(){
   // simple ascii square
   cout << "\n";
   cout << "    *********\n";
   cout << "    *       *\n";
   cout << "    *       *\n";
   cout << "    *       *\n";
   cout << "    *       *\n";
   cout << "    *********\n";
   cout << "\n";
}
extern "C" {
shape *maker(){
   return new square;
}
class proxy { public:
   proxy(){
      // register the maker with the factory
      factory["square"] = maker;
   }
};
// our one instance of the proxy
proxy p;
}
