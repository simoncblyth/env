#ifndef G4DAEHITLIST_H
#define G4DAEHITLIST_H

#include <vector>
#include <string>
#include <G4ThreeVector.hh>

#include "G4DAEChroma/G4DAEHit.hh"

class G4DAEArray ;

class G4DAEHitList {

public:
  G4DAEHitList( G4DAEArray* array );
  G4DAEHitList( std::size_t itemcapacity = 0, float* data = NULL);
  virtual ~G4DAEHitList();

public:
  void AddHit( G4DAEHit& hit );
  void Print(const char* msg="G4DAEHitList::Print") const ; 

public:
  // other  
  static std::string GetPath( const char* evt="dummy" , const char* tmpl="DAEHIT_PATH_TEMPLATE");   
  static G4DAEHitList* Load(const char* evt="1", const char* key="GPL", const char* tmpl="DAEHIT_PATH_TEMPLATE" );
  void Save(const char* evt="dummy", const char* key="GPL", const char* tmpl="DAEHIT_PATH_TEMPLATE" );

  static G4DAEHitList* LoadPath(const char* path, const char* key="GPL");
  void SavePath(const char* path, const char* key="GPL");

private:
   G4DAEArray* m_array ;


};

#endif

