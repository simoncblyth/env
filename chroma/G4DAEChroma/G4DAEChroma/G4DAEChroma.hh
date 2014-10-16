/*
*/
#ifndef G4DAECHROMA_H
#define G4DAECHROMA_H 1

#include <cstddef>
#include "G4ThreeVector.hh"
#include "G4Track.hh"
#include "G4AffineTransform.hh"

class ZMQRoot ; 
class ChromaPhotonList ;
class G4DAEGeometry ;
 
class G4DAEChroma 
{
public:
    struct Hit {
        // global
        G4ThreeVector gpos ;
        G4ThreeVector gdir ;
        G4ThreeVector gpol ;

       // local : maybe just keep local, inplace transform ?
        G4ThreeVector lpos ;
        G4ThreeVector ldir ;
        G4ThreeVector lpol ;

        float t ;
        float wavelength ;
        int   hitindex ; 
        int   pmtid ;
        int   volumeindex ;

        void LocalTransform(G4AffineTransform& trans)
        { 
            lpos = trans.TransformPoint(gpos);
            lpol = trans.TransformAxis(gpol);
            lpol = lpol.unit();
            ldir = trans.TransformAxis(gdir);
            ldir = ldir.unit();
        }
        void Print(){
              G4cout 
                     << " hitindex " << hitindex 
                     << " volumeindex " << volumeindex 
                     << " pmtid "       << pmtid 
                     << " t "     << t 
                     << " wavelength " << wavelength 
                     << " gpos "  << gpos 
                     << " gdir "  << gdir 
                     << " gpol "  << gpol 
                     << G4endl ; 
         }
    }; 

public:
    static G4DAEChroma* GetG4DAEChroma();
    static G4DAEChroma* GetG4DAEChromaIfExists();
protected:
    G4DAEChroma();
public:
    virtual ~G4DAEChroma();

    void SetGeometry(G4DAEGeometry* geo);
    G4DAEGeometry* GetGeometry();

    void ClearAll();
    void CollectPhoton(const G4Track* aPhoton );
    void Propagate(G4int batch_id);
    bool ProcessHit( const ChromaPhotonList* cpl, std::size_t index );
 
private:
  // Singleton instance
  static G4DAEChroma* fG4DAEChroma;

  // ZeroMQ network socket utility 
  ZMQRoot* fZMQRoot ; 

  // transport ready TObject 
  ChromaPhotonList* fPhotonList ; 

  // test receiving object from remote zmq server
  ChromaPhotonList* fPhotonList2 ; 

  // Geometry Transform cache, used to convert global to local coordinates
  G4DAEGeometry* fGeometry ; 

};


#endif

 
