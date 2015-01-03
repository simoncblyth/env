#ifndef G4DAECOLLECTOR_H
#define G4DAECOLLECTOR_H 1

#include "G4CollectionNameVector.hh"
#include "G4DAEChroma/G4DAEHit.hh"
class G4DAEArrayHolder ; 
#include <vector>
#include <cstddef>
#include <string>

class G4HCofThisEvent ;
class G4VHitsCollection ;

class G4DAETransformCache ;

//  **G4DAECollector**
//
//  Generic hit collector base class with  
//  pure virtual methods which must be implemented 
//  in specific detector subclasses.
//


#ifdef DEBUG_HITLIST
class G4DAEHitList ; 
#endif

class G4DAECollector  {

public:
    typedef std::vector<std::size_t> IDVec ; 

    G4DAECollector();
    virtual ~G4DAECollector();

    virtual void DefineCollectionNames(G4CollectionNameVector&) = 0;
    virtual void CreateHitCollections( const std::string& sdname, G4HCofThisEvent* hce ) = 0;
    virtual void StealHitCollections( const std::string& target,  G4HCofThisEvent* hce ) = 0;
    virtual void Collect( const G4DAEHit& hit ) = 0;

    virtual void AddSomeFakeHits(const IDVec& sensor_ids);
    virtual void CollectHits( G4DAEArrayHolder* photons, G4DAETransformCache* cache );

public:
    static void DumpStatistics( G4HCofThisEvent* hce, int detail=0 );
    static void DumpHC( G4VHitsCollection* hc,  int index, int detail );


public:
#ifdef DEBUG_HITLIST
    G4DAEHitList* GetHits();
#endif

private:

#ifdef DEBUG_HITLIST
    G4DAEHitList* m_hits ;  // for debugging 
#endif


};

#endif

