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

    //printf("mocknuwa _batch  %s => %d : %d  \n", _batch, batch_id[0], batch_id[1] ); 
    //printf("mocknuwa _ctrl   %s => %d : %d  \n", _ctrl, ctrl_id[0], ctrl_id[1] ); 
    //printf("mocknuwa _range %s => %d : %d  \n", _range, range[0], range[1] ); 


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

    Map_t empty ;

    for(int cid=ctrl_id[0] ; cid < ctrl_id[1] ; cid++ ){
    for(int bid=batch_id[0] ; bid < batch_id[1] ; bid++ ){

            Map_t ctrl   = database ? database->GetOne("select * from ctrl  where id=? ;", cid ) : empty ; 
            Map_t batch  = database ? database->GetOne("select path, tag from batch where id=? ;", bid ) : empty ; 

            Map_t args ;
            args["COLUMNS"] = "ctrl_id:i,batch_id:i,hit:i,dphotons:s,nphotons:i";
            args["ctrl_id"] = toStr<int>(cid) ; 
            args["batch_id"] = toStr<int>(bid) ; 
            args["hit"] = toStr<int>(0) ;   // 1:reply with only hits, 0:reply with all 


            G4DAEMetadata* phometa = new G4DAEMetadata("{}") ;
            phometa->AddMap("ctrl", ctrl);
            phometa->AddMap("batch", batch);
            phometa->AddMap("args", args);


            const char* path = batch["path"].c_str();
            G4DAEPhotons* all = G4DAEPhotons::LoadPath( path );
            G4DAEPhotons* photons = all->Slice(range[0], range[1]);

            args["COLUMNS"] += "dphotons:s,aphotons:i,nphotons:i,arange:i,brange:i";
            args["dphotons"] = photons->GetDigest() ;
            args["aphotons"] = toStr<int>(all->GetCount()) ;
            args["nphotons"] = toStr<int>(photons->GetCount()) ;
            args["arange"]   = toStr<int>(range[0]) ;
            args["brange"]   = toStr<int>(range[1]) ;

            delete all ;

            phometa->Print("#phometa"); 
            photons->AddLink(phometa);

            G4DAEPhotons* hits = chroma->Propagate(photons);   // propagation + hit collection


            G4DAEMetadata* hitmeta = hits->GetLink();  

            Map_t mhits ; 
            mhits["COLUMNS"] = "dhits:s,nhits:i,std:i,stddt:s,loc:i,locdt:s";
            mhits["dhits"] = hits->GetDigest();
            mhits["nhits"] = toStr<int>(hits->GetCount());
            mhits["std"]   = now("%s", 20, 1 );
            mhits["stddt"] = now("%Y-%m-%d %H:%M:%S", 20, 1);
            mhits["loc"]   = now("%s", 20, 0 );
            mhits["locdt"] = now("%Y-%m-%d %H:%M:%S", 20, 0);

            
            hitmeta->AddMap("mhits", mhits);         
            
            std::string logfield = "ctrl_id,batch_id,tottime,nwork,std,stddt,loc,locdt,in_vms,out_vms" ;
            int log_id = database ? database->Insert(hitmeta, "log", logfield.c_str() ) : 0 ; 

            std::string timestamp = now("%Y%m%d_%H%M%S", 20, 0);

            std::vector<std::string> elem ;
            elem.push_back(basepath(path,'.'));
            elem.push_back("v001");
            elem.push_back(timestamp);
            elem.push_back("mocknuwa.json");
            std::string logpath = join(elem, '/'); 
            
            Map_t mlog ;
            mlog["logfield"] = logfield ;
            mlog["log_id"] = toStr<int>(log_id); 
            mlog["logpath"] = logpath ;

            hitmeta->AddMap("mlog", mlog); 
            hitmeta->Print("#hitmeta");
            hitmeta->PrintToFile(logpath.c_str()); 


    } // bid
    } // cid

    return 0 ; 
}




