#ifndef G4DAECOMMON_H 
#define G4DAECOMMON_H 

#include <string>
#include <vector>

class G4AffineTransform ;

std::string transform_rep( G4AffineTransform& transform );
void split( std::vector<std::string>& elem, const char* linekey, char delim );


#endif

