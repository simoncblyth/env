#include "G4DAEChroma/G4DAECommon.hh"

#include <sstream>
#include "G4AffineTransform.hh"

#include "md5digest.h"

using namespace std ; 

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



void split( vector<string>& elem, const char* linekey, char delim )
{
    const char* line = getenv(linekey);
    if(line == NULL){ 
        cout << "split envvar not defined : " << linekey << endl ; 
        return ;
    }   
    istringstream f(line);
    string s;
    while (getline(f, s, delim)) elem.push_back(s);
}




