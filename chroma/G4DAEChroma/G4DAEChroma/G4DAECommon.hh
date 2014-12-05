#ifndef G4DAECOMMON_H 
#define G4DAECOMMON_H 

#include <string>
#include <sstream>
#include <vector>


// using union for co-location of int, unsigned int or float within "float" slots 
typedef union {
    float f ;
    int i ;
    unsigned int u ;
} uif_t ;  


class G4AffineTransform ;

struct zmq_msg_t ;

std::string transform_rep( G4AffineTransform& transform );
void split( std::vector<std::string>& elem, const char* line, char delim );
void isplit( std::vector<int>& elem, const char* line, char delim );


void getintpair( const char* range, char delim, int* a, int* b );




std::string md5digest( const char* str, int length );
void DumpBuffer(const char* buffer, std::size_t buflen, std::size_t maxlines=64); 
void DumpVector(const std::vector<float>& v, std::size_t itemsize); 

extern int b_recv( void* socket, zmq_msg_t& msg );
extern int b_send( void* socket, const char* bytes, size_t size, int flags=0 );
extern int s_send (void *socket, char *str); 
extern char* s_recv (void *socket); 

void current_time(char* buf, int buflen, const char* tfmt, int utc);
std::string now(const char* tfmt, const int buflen, int utc);


template<typename T>
std::string toStr(const T& value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}




#endif

