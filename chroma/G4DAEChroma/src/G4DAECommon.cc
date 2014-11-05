#include "G4DAEChroma/G4DAECommon.hh"

#include <sstream>
#include "G4AffineTransform.hh"

#include "md5digest.h"

using namespace std ; 


void DumpBuffer(const char* buffer, size_t buflen) 
{
   const char* hfmt = "\n%04X : " ;
   for (int i = 0; i < buflen ; i++){
       if(i % 16 == 0) printf(hfmt, i ); 
       printf("%02X ", buffer[i] & 0xff );
   }
   printf(hfmt, buflen );
   printf("\n"); 
}


void DumpVector(const std::vector<float>& v, size_t itemsize) 
{
   const char* hfmt = "\n%04d : " ;
   for (int i = 0; i < v.size() ; i++){
       if(i % itemsize == 0) printf(hfmt, i ); 
       printf("%10.3f ", v[i]);
   }
   printf(hfmt, v.size() );
   printf("\n"); 
}






string md5digest( const char* buffer, int len )
{
    char* out = md5digest_str2md5(buffer, len);
    string digest(out); 
    free(out);
    return digest;
}

string transform_rep( G4AffineTransform& transform )
{

   G4RotationMatrix rotation = transform.NetRotation();
   G4ThreeVector rowX = rotation.rowX();
   G4ThreeVector rowY = rotation.rowY();
   G4ThreeVector rowZ = rotation.rowZ();
   G4ThreeVector tran = transform.NetTranslation(); 
   
   stringstream ss; 
   ss << tran << " " << rowX << rowY << rowZ  ;
   return ss.str();
}



void split( vector<string>& elem, const char* line, char delim )
{
    if(line == NULL){ 
        cout << "split NULL line not defined : " << endl ; 
        return ;
    }   
    istringstream f(line);
    string s;
    while (getline(f, s, delim)) elem.push_back(s);
}

void isplit( vector<int>& elem, const char* line, char delim )
{
    if(line == NULL){ 
        cout << "isplit NULL line not defined : " << endl ; 
        return ;
    }   
    istringstream f(line);
    string s;
    while (getline(f, s, delim)) elem.push_back(atoi(s.c_str()));
}




