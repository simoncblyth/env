#include "RapSqlite/Common.hh"

void split( std::vector<std::string>& elem, const char* line, char delim )
{
    if(line == NULL) return ;
    std::istringstream f(line);
    std::string s;
    while (getline(f, s, delim)) elem.push_back(s);
}


Map_t dsplit( const char* spec, char adelim, char bdelim )
{
    Map_t map ; 
    Vec_t elem;
    split(elem, spec, adelim);
    for(std::size_t i=0 ; i<elem.size() ; ++i )
    {
       Vec_t pair ;
       split(pair, elem[i].c_str(), bdelim);
       //printf("elem i %lu [%s] \n", i, elem[i].c_str());

       assert(pair.size() == 2 );
       map[pair[0]] = pair[1];
    } 
    return map;
}



