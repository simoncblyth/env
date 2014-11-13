#ifndef G4DAEHITLIST_H
#define G4DAEHITLIST_H

#include <vector>
#include <string>
#include <G4ThreeVector.hh>

#include "G4DAEChroma/G4DAEHit.hh"
#include "G4DAEChroma/G4DAEArrayHolder.hh"

class G4DAEArray ;

class G4DAEHitList : public G4DAEArrayHolder {

  static const char* TMPL ;   // name of envvar containing path template 
  static const char* SHAPE ;  // numpy array itemshape eg "8,3" or "4,4" 
  static const char* KEY ;  

public:
  G4DAEHitList( G4DAEArray* array );
  G4DAEHitList( std::size_t itemcapacity = 0, float* data = NULL);
  virtual ~G4DAEHitList();

public:
  void AddHit( G4DAEHit& hit );


// the below cannot go to base due to the static tmpl arguments
public:
  static std::string GetPath( const char* evt, const char* tmpl=TMPL);   
  static G4DAEHitList* Load(const char* evt, const char* key=KEY, const char* tmpl=TMPL);
  static G4DAEHitList* LoadPath(const char* path, const char* key=KEY);

public:
  virtual void Save(const char* evt, const char* key=KEY, const char* tmpl=TMPL );
  virtual void SavePath(const char* path, const char* key=KEY);

};

#endif

