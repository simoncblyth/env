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




