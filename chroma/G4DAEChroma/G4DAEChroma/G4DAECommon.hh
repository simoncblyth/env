#ifndef G4DAECOMMON_H 
#define G4DAECOMMON_H 

#include <string>
#include <vector>

class G4AffineTransform ;

std::string transform_rep( G4AffineTransform& transform );
void split( std::vector<std::string>& elem, const char* line, char delim );
void isplit( std::vector<int>& elem, const char* line, char delim );

std::string md5digest( const char* str, int length );
void DumpBuffer(const char* buffer, std::size_t buflen);
void DumpVector(const std::vector<float>& v, std::size_t itemsize); 


#endif

