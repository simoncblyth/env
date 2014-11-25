#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAETransport.hh"
#include "G4DAEChroma/G4DAECollector.hh"
#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/G4DAETransformCache.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#include "G4DAEChroma/G4DAEHitList.hh"
#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAEDatabase.hh"
#include "G4DAEChroma/G4DAEPhotonList.hh"


#include "DybG4DAECollector.h"

#include "G4SDManager.hh"

class ITouchableToDetectorElement ;


using namespace std ;
#include <iostream>

#define NOT_NUWA 1
#include "DsChromaRunAction_BeginOfRunAction.icc"


void Mockup_DetDesc_SD()
{
   G4SDManager* SDMan = G4SDManager::GetSDMpointer();
   //SDMan->SetVerboseLevel( 10 );

   G4DAESensDet* sd1 = new G4DAESensDet("DsPmtSensDet","");
   sd1->SetCollector(new DybG4DAECollector );  
   sd1->initialize();
   SDMan->AddNewDetector( sd1 );

   G4DAESensDet* sd2 = new G4DAESensDet("DsRpcSensDet","");
   sd2->SetCollector(new DybG4DAECollector );  
   sd2->initialize();
   SDMan->AddNewDetector( sd2 );
}



G4DAEPhotons* MockPhotonList(G4DAETransformCache* cache, std::size_t size)
{
    G4DAEPhotons* photons = (G4DAEPhotons*)new G4DAEPhotonList(size);

    G4ThreeVector lpos(0,0,1500) ;
    G4ThreeVector lpot(0,0,1000) ; 
    G4ThreeVector lpol(0,0,1) ; 
    const float time = 1. ;
    const float wavelength = 550. ;

    size_t count = 0 ;
    for( size_t index = 0 ; index < cache->GetSize() ; ++index ) // cache contains affine transforms for all PMTs
    {
        if( index % 1 == 0 && count < size )
        {
            int pmtid = cache->GetKey(index);

            G4AffineTransform* pg2l = cache->GetSensorTransform(pmtid);
            assert(pg2l);

            G4AffineTransform g2l(*pg2l);
            G4AffineTransform l2g(g2l.Inverse());
           
            G4ThreeVector gpos(l2g.TransformPoint(lpos));
            G4ThreeVector gpot(l2g.TransformPoint(lpot));
            G4ThreeVector gdir = gpot - gpos ;
            G4ThreeVector gpol(l2g.TransformAxis(lpol));

            //bool vertical = ( gdir.x() == 0. && gdir.y() == 0. ); // problematic vertical photons
            photons->AddPhoton( gpos, gdir, gpol, time, wavelength, pmtid );
            count++ ;
        }
    } 

    return photons ; 
}


void getintpair( const char* envvar, char delim, int* a, int* b )
{
    const char* val = getenv(envvar);
    if(!val) return ;

    std::vector<std::string> elem ;  
    split(elem, val, delim);
    assert( elem.size() == 2 );

    *a = atoi(elem[0].c_str()) ;
    *b = atoi(elem[1].c_str()) ;
}


int main(int argc, const char** argv)
{
    //
    //  TODO: move to arguments specifiying one or more 
    //        primary keys of config and photondata tables in DB
    // 
    //        DB select the specified entries as Map_t, these
    //        are converteted to JSON for transport to propagator
    //

    const char* name = NULL ; 
    const char* tag  = "hh" ; // eg "hv" for vbo prop, "ha" for array (non-vbo) prop
    if(argc > 1) name = argv[1] ; 
    if(argc > 2) tag  = argv[2] ; 


    if( name == NULL )
    {
       printf("expecting an argument specifying the photons to mockup or load \n");
       exit(1);
    }
 
    string htag(tag) ;
    htag += name ; 


    Mockup_DetDesc_SD();

    DsChromaRunAction_BeginOfRunAction(
         "G4DAECHROMA_CLIENT_CONFIG", 
         "G4DAECHROMA_CACHE_DIR", 
         "DsPmtSensDet" , 
         "G4DAECHROMA_DATABASE_PATH", 
          NULL, 
          "" ); // config 

    G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();

    // TODO: simplify API, so normal operations all done through "chroma" instance ?
    //       reposition photons and hits outside transport 
    //

    G4DAETransport*   transport = chroma->GetTransport();
    G4DAEDatabase*    database = chroma->GetDatabase();
    G4DAETransformCache*  cache = chroma->GetTransformCache();


    G4HCofThisEvent* HCE = G4SDManager::GetSDMpointer()->PrepareNewEvent();  // calls Initialize for registered SD 


    // TODO: arrange config tables to have convenient PKs for selection based on command line args
    const char* sql = "select max_blocks, max_steps, nthreads_per_block threads_per_block, seed, reset_rng_states from mocknuwa limit 1 ;" ;
    int nrow = database->Query(sql);
    Map_t row = database->GetRow(0);
    if(row.size() < 2 ){
        printf("mocknuwa: DB query error OR empty result OR row too small nrow %d sql: %s \n", nrow, sql );
        exit(1);
    }

    G4DAEMetadata* ctrl = new G4DAEMetadata("{}");
    ctrl->AddMap("ctrl", row );
    ctrl->Set("htag", htag.c_str());
    ctrl->Set("hit", "0");
    ctrl->Merge("args");
    ctrl->Print(); 


    G4DAEPhotons* all = NULL ;
    if( strcmp(name,"MOCK") == 0 )
    {
        printf("mocknuwa: generating photon list with MockPhotonList\n");
        all = MockPhotonList( cache, cache->GetSize() );
    }
    else
    {
        printf("mocknuwa: loading photon list named %s\n", name);
        all = G4DAEPhotons::Load(name); 
    } 

    assert(all);

    int a = 0 ;
    int b = 0 ;
    getintpair( "RANGE", ':', &a, &b ); // python style 0:1 => [0]   0:0 means ALL
    G4DAEPhotons* photons = (G4DAEPhotons*)new G4DAEPhotonList(all, a, b);

    photons->AddLink(ctrl);


    photons->Print("mocknuwa: photons"); 
    //photons->Details(0); 

    transport->SetPhotons( photons );

    chroma->Propagate(1); // PropagateToHits : <1  fakes the propagation, ie just passes all photons off as hits

    G4DAEPhotons* hits = transport->GetHits(); assert(hits);
    hits->Print("mocknuwa: hits___");

    G4DAEPhotons::Save(hits, htag.c_str());
    hits->Print("after save");



    G4DAESensDet* sd = chroma->GetSensDet();
    sd->EndOfEvent(HCE); // G4 calls this for hit handling?

    G4DAEHitList* hitlist = sd->GetCollector()->GetHits(); 
    if( hitlist )
    {
        hitlist->Save( htag.c_str() );
        hitlist->Print("mocknuwa: hitlist");
    }
    else
    {
        printf("null hitlist\n");
    }


    G4DAEMetadata* meta = hits->GetLink();
    if( meta )
    {
        meta->Set("htag",     htag.c_str() );
        meta->Set("dphotons", photons->GetDigest().c_str() );
        meta->Set("dhits",    hits->GetDigest().c_str() );

        if( hitlist ) meta->Set("dhitlist", hitlist->GetDigest().c_str() );

        meta->Set("COLUMNS",  "htag:s,dphotons:s,dhits:s,dhitlist:s");
        meta->Merge("caller");  // add "caller" object with these settings to JSON tree

        meta->Print();
        meta->PrintToFile("/tmp/mocknuwa.json");

        meta->SetName("mocknuwa");   // DB tablename
        database->Insert(meta); 
    }
    else
    {
        printf("mocknuwa: meta NULL \n");
    }


    //
    // TODO: check can extract hitlists from actual collections
    //       to make connection to standard Geant4 propagation hits 
    //

    return 0 ; 
}




