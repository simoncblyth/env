#ifndef CAPTUREMAP_H
#define CAPTUREMAP_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>

using namespace std;

typedef map<string, string> Map;
typedef pair<string, string> Pair;

class CaptureMap {
   private :
      Map m ;
   public :
      void Add( string k , string v );
      string Get( string k ){
         return m[k] ;
      }
};


#endif
