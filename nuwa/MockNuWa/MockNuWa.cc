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


using namespace std ;
#include <iostream>
#include <sstream>

class ITouchableToDetectorElement ;

#define NOT_NUWA 1
#include "DsChromaRunAction_BeginOfRunAction.icc"



template<typename T>
std::string toStr(const T& value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}



int main(int argc, const char** argv)
{
    // python style ranges 
    const char* _batch  = "1:2" ; 
    const char* _ctrl = "1:2" ;           
    const char* _range  = getenv("RANGE") ;

    if(argc > 1) _batch   = argv[1] ; 
    if(argc > 2) _ctrl  = argv[2] ; 

    int batch_id[2] ;
    int ctrl_id[2] ;
    int range[2] = {0} ;

    getintpair( _batch,  ':', batch_id, batch_id+1 );
    getintpair( _ctrl, ':', ctrl_id, ctrl_id+1 );
    getintpair( _range, ':', range, range+1 ); 

    printf("mocknuwa _batch  %s => %d : %d  \n", _batch, batch_id[0], batch_id[1] ); 
    printf("mocknuwa _ctrl   %s => %d : %d  \n", _ctrl, ctrl_id[0], ctrl_id[1] ); 
    printf("mocknuwa _range %s => %d : %d  \n", _range, range[0], range[1] ); 


    // setup Geant4 SDs like NuWa/DetDesc does

    G4DAESensDet::MockupSD("DsPmtSensDet", new DybG4DAECollector );
    G4DAESensDet::MockupSD("DsRpcSensDet", new DybG4DAECollector );

    // initializing G4DAEChroma, including hooking up trojan SD

    DsChromaRunAction_BeginOfRunAction(
         "G4DAECHROMA_CLIENT_CONFIG", 
         "G4DAECHROMA_CACHE_DIR", 
         "DsPmtSensDet" , 
         "G4DAECHROMA_DATABASE_PATH", 
          NULL, 
          "" ); 


    G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
    G4DAEDatabase*    database = chroma->GetDatabase();

    G4HCofThisEvent* HCE = G4SDManager::GetSDMpointer()->PrepareNewEvent();  // calls Initialize for registered SD 


    for(int cid=ctrl_id[0] ; cid < ctrl_id[1] ; cid++ )
    {
        Map_t ctrl  = database->GetOne("select * from ctrl  where id=? ;", cid ); assert(!ctrl.empty()); 

        for(int bid=batch_id[0] ; bid < batch_id[1] ; bid++ )
        {
            // prepare photon data and link metadata

            Map_t batch  = database->GetOne("select id batch_id, path, tag from batch where id=? ;", bid ); 
            assert(!batch.empty()); 
            std::string tag = batch["tag"];

            G4DAEPhotons* all = G4DAEPhotons::LoadPath( batch["path"].c_str() );
            G4DAEPhotons* photons = all->Slice(range[0], range[1]);
            delete all ;

            Map_t args ;
            args["COLUMNS"] = "config_id:i,batch_id:i,tag:s,hit:i,dphotons:s,nphotons:i";
            args["ctrl_id"] = toStr<int>(cid) ; 
            args["batch_id"] = toStr<int>(bid) ; 

            args["hit"] = toStr<int>(0) ;   // 1:reply with only hits, 0:reply with all 
            args["dphotons"] = photons->GetDigest() ;
            args["nphotons"] = toStr<int>(photons->GetCount()) ;


            G4DAEMetadata* req = new G4DAEMetadata("{}") ;
            req->AddMap("ctrl", ctrl);
            req->AddMap("batch", batch);
            req->AddMap("args", args);
            req->Print(); 
            photons->AddLink(req);

            // doing the propagation 

            G4DAEPhotons* hits = chroma->Propagate(bid, photons); 

            G4DAEMetadata* rep = hits->GetLink();  
            hits->Print("mocknuwa: hits___");

            Map_t mhits ; 
            mhits["dhits"] = hits->GetDigest();
            mhits["nhits"] = toStr<int>(hits->GetCount());
            rep->AddMap("mhits", mhits); 

            //rep->Set("COLUMNS",  "config_id:i,batch_id:i,dphotons:s,dhits:s");
            //rep->Set("config_id", cid);
            //rep->Set("batch_id",  bid);
            //rep->Set("dhits",     hits->GetDigest().c_str() );
            //rep->Merge("caller");  // add "caller" object with these digests to JSON tree

            rep->Print();
            rep->PrintToFile("/tmp/mocknuwa.json");  
            // write with a timestamp (and the rowid of the insert) 


            int stat_id = database->Insert(rep, "stat", "config_id,batch_id,tottime,nwork" ); 
            printf("stat insert %d \n", stat_id);

            //TODO: insert datetime column



        } // bid
    }     // cid

    return 0 ; 
}




