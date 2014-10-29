#ifndef G4DAECOLLECTOR_H
#define G4DAECOLLECTOR_H 1

#include "G4CollectionNameVector.hh"
#include "G4DAEChroma/G4DAEHit.hh"
#include <vector>
#include <cstddef>
#include <string>

class G4HCofThisEvent ;
class ChromaPhotonList ; 
class G4DAEGeometry ;

//  **G4DAECollector**
//
//  Generic hit collector base class with  
//  pure virtual methods which must be implemented 
//  in specific detector subclasses.
//

class G4DAECollector  {

public:
    typedef std::vector<std::size_t> IDVec ; 

    G4DAECollector(){};
    virtual ~G4DAECollector(){};

    virtual void DefineCollectionNames(G4CollectionNameVector&) = 0;
    virtual void CreateHitCollections( const std::string& sdname, G4HCofThisEvent* hce ) = 0;
    virtual void StealHitCollections( const std::string& target,  G4HCofThisEvent* hce ) = 0;
    virtual void Collect( const G4DAEHit& hit ) = 0;

    virtual void AddSomeFakeHits(const IDVec& sensor_ids);
    virtual void DumpStatistics( G4HCofThisEvent* hce );
    virtual void CollectHits( ChromaPhotonList* cpl, G4DAEGeometry* geometry );


};

#endif

