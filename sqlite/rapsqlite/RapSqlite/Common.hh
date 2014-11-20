#ifndef RSCOMMON_H
#define RSCOMMON_H

#include <string>
#include <vector>
#include <map>

#include <stdio.h>
#include <cstdlib>
#include <sqlite3.h> 

#include <cassert>
#include <sstream>

class Table ; 

typedef std::vector<std::string> Vec_t ; 
typedef std::map<std::string,std::string> Map_t ; 
typedef std::vector<Map_t> VMap_t ; 
typedef std::map<std::string,Table*> TableMap_t ; 

extern void split( std::vector<std::string>& elem, const char* line, char delim );
extern Map_t dsplit( const char* spec, char adelim, char bdelim );


#endif

