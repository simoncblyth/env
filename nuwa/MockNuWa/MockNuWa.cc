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
    G4ThreeVector ldir(0,0,-1) ;
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
            G4ThreeVector gdir(l2g.TransformAxis(ldir));
            G4ThreeVector gpol(l2g.TransformAxis(lpol));

            photons->AddPhoton( gpos, gdir, gpol, time, wavelength, pmtid );
            count++ ;
        }
    } 

    return photons ; 
}


void getintpair( const char* range, char delim, int* a, int* b )
{
    if(!range) return ;

    std::vector<std::string> elem ;  
    split(elem, range, delim);
    assert( elem.size() == 2 );

    *a = atoi(elem[0].c_str()) ;
    *b = atoi(elem[1].c_str()) ;
}



G4DAEPhotons* prepare_photons(const char* tag, G4DAETransformCache* cache)
{
    G4DAEPhotons* all = NULL ;
    if( strcmp(tag,"MOCK") == 0 )
    {
        printf("mocknuwa: generating photon list with MockPhotonList\n");
        all = MockPhotonList( cache, cache->GetSize() );
    }
    else
    {
        printf("mocknuwa: loading photon list named %s\n", tag);
        all = G4DAEPhotons::Load(tag); 
    } 
    assert(all);

    int a = 0 ;
    int b = 0 ;
    getintpair(getenv("RANGE"), ':', &a, &b ); // python style 0:1 => [0]   0:0 means ALL
    G4DAEPhotons* photons = (G4DAEPhotons*)new G4DAEPhotonList(all, a, b);
    return photons ; 
}


int main(int argc, const char** argv)
{
    const char* _batch  = "1:2" ; 
    const char* _config = "1:2" ;  // python style range 

    if(argc > 1) _batch   = argv[1] ; 
    if(argc > 2) _config  = argv[2] ; 

    int batch_id[2] ;
    int config_id[2] ;
    getintpair( _batch,  ':', batch_id, batch_id+1 );
    getintpair( _config, ':', config_id, config_id+1 );

    printf("mocknuwa _batch  %s => %d : %d  \n", _batch, batch_id[0], batch_id[1] ); 
    printf("mocknuwa _config %s => %d : %d  \n", _config, config_id[0], config_id[1] ); 


    Mockup_DetDesc_SD();

    DsChromaRunAction_BeginOfRunAction(
         "G4DAECHROMA_CLIENT_CONFIG", 
         "G4DAECHROMA_CACHE_DIR", 
         "DsPmtSensDet" , 
         "G4DAECHROMA_DATABASE_PATH", 
          NULL, 
          "" ); // config 

    G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
    G4DAEDatabase*    database = chroma->GetDatabase();
    G4DAETransformCache*  cache = chroma->GetTransformCache();

    G4HCofThisEvent* HCE = G4SDManager::GetSDMpointer()->PrepareNewEvent();  // calls Initialize for registered SD 

    for(int cid=config_id[0] ; cid < config_id[1] ; cid++ )
    {
        // prepare config
        Map_t config  = database->GetOne("select * from config  where id=? ;", cid ); assert(!config.empty()); 
        G4DAEMetadata* req = new G4DAEMetadata(config, "ctrl");

        for(int bid=batch_id[0] ; bid < batch_id[1] ; bid++ )
        {

            // prepare photon data
            Map_t batch  = database->GetOne("select * from batch  where id=? ;", bid ); 
            assert(!batch.empty()); 
            const char* tag = batch[std::string("tag")].c_str();

            G4DAEPhotons* photons = prepare_photons(tag, cache);

            req->SetKV("args", "COLUMNS", "config_id:i,batch_id:i,tag:s,hit:i,dphotons:s");
            req->SetKV("args", "config_id", cid );
            req->SetKV("args", "batch_id",  bid );
            req->SetKV("args", "tag",  tag );
            req->SetKV("args", "hit",  0 );
            req->SetKV("args", "dphotons",  photons->GetDigest().c_str() );
            req->Print(); 

            photons->Print("mocknuwa: photons"); 
            photons->AddLink(req);

            G4DAEPhotons* hits = chroma->Propagate(bid, photons); assert(hits);  
            hits->Print("mocknuwa: hits___");

            // TODO:consistency asserts on response metadata

            G4DAEMetadata* rep = hits->GetLink();  assert(rep);

            std::string check_cid = rep->Get("args", "config_id");
            printf("check_cid %s \n", check_cid.c_str());

            rep->Set("COLUMNS",  "config_id:i,batch_id:i,dphotons:s,dhits:s");
            rep->Set("config_id", cid);
            rep->Set("batch_id",  bid);
            rep->Set("dhits",     hits->GetDigest().c_str() );
            rep->Merge("caller");  // add "caller" object with these digests to JSON tree

            rep->PrintMap("hits js map");
            rep->Print();
            rep->PrintToFile("/tmp/mocknuwa.json");  // write with a timestamp (and the rowid of the insert) 

            int stat_id = database->Insert(rep, "stat", "config_id,batch_id,tottime,nwork" ); 
            printf("stat insert %d \n", stat_id);

            //TODO: insert datetime column
            //



        } // bid
    }     // cid

    return 0 ; 
}




