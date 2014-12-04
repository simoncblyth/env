#ifndef G4DAEMAP_H
#define G4DAEMAP_H

#include <string>
#include <map>

typedef std::map<std::string,std::string>  Map_t ;

class G4DAEMap {
public:
    G4DAEMap(Map_t& map);
    virtual ~G4DAEMap();
private:
    Map_t m_map ; 

};


#endif
