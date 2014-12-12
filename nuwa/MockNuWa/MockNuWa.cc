#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAETransport.hh"
#include "G4DAEChroma/G4DAECollector.hh"
#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/G4DAETransformCache.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#include "G4DAEChroma/G4DAETime.hh"
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




void Propagate(int seq, int bid, int cid, int* range )
{
    G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
    G4DAEDatabase* db = chroma->GetDatabase();
    G4HCofThisEvent* HCE = G4SDManager::GetSDMpointer()->PrepareNewEvent();  // calls Initialize for registered SD 

    Map_t empty ;
    Map_t ctrl   = db ? db->GetOne("select * from ctrl  where id=? ;", cid ) : empty ; 
    Map_t batch  = db ? db->GetOne("select path, tag from batch where id=? ;", bid ) : empty ; 

    Map_t args ;
    args["COLUMNS"] = "seq:i,ctrl_id:i,batch_id:i,hit:i,dphotons:s,nphotons:i";
    args["seq"] = toStr<int>(seq) ; 
    args["ctrl_id"] = toStr<int>(cid) ; 
    args["batch_id"] = toStr<int>(bid) ; 
    args["hit"] = toStr<int>(1) ;   // 1:reply with only hits, 0:reply with all 


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


    G4DAEPhotons* hits = NULL ;
    double t_propagate = -1. ; 
    {
        double t0 = getRealTime();
        hits = chroma->Propagate(photons);   // propagation + hit collection
        t_propagate = getRealTime() - t0 ;
    }
    G4DAEMetadata* hitmeta = hits->GetLink();  

    Map_t mhits ; 
    mhits["COLUMNS"] = "dhits:s,nhits:i,std:i,stddt:s,loc:i,locdt:s";
    mhits["dhits"] = hits->GetDigest();
    mhits["nhits"] = toStr<int>(hits->GetCount());
    mhits["std"]   = now("%s", 20, 1 );
    mhits["stddt"] = now("%Y-%m-%d %H:%M:%S", 20, 1);
    mhits["loc"]   = now("%s", 20, 0 );
    mhits["locdt"] = now("%Y-%m-%d %H:%M:%S", 20, 0);
    mhits["tprop"] = toStr<double>(t_propagate);


    hitmeta->AddMap("mhits", mhits);         
    
    std::string logfield = "seq,ctrl_id,batch_id,tottime,nwork,loc,locdt,in_vms,out_vms,in_rss,out_rss" ;
    int log_id = db ? db->Insert(hitmeta, "log", logfield.c_str() ) : 0 ; 

    std::string timestamp = now("%Y%m%d_%H%M%S", 20, 0);

    std::vector<std::string> elem ;
    elem.push_back(basepath(path,'.'));
    elem.push_back("scan");
    elem.push_back(toStr<int>(cid));
    elem.push_back("mocknuwa.json");
    std::string logpath = join(elem, '/'); 
    
    Map_t mlog ;
    mlog["logfield"] = logfield ;
    mlog["log_id"] = toStr<int>(log_id); 
    mlog["logpath"] = logpath ;

    hitmeta->AddMap("mlog", mlog); 
    hitmeta->Print("#hitmeta");
    hitmeta->PrintToFile(logpath.c_str()); 

}





std::vector<long> getivec( G4DAEDatabase* db, const char* arg )
{
    std::vector<long> ivec ;
    if( arg[0] == 's' )       // argument beginning with 's'elect 
    {
        ivec = db->GetIVec("id", arg);
    } 
    else 
    {                         // colon delimited integer range 1:2 
        int _id[2]; 
        getintpair( arg,  ':', _id, _id+1 );
        for(long i=_id[0] ; i<_id[1] ; i++) ivec.push_back(i);
    }
    return ivec ;
}


int main(int argc, const char** argv)
{
    const char* _batch = "select id from batch where nwork>2000 order by id;" ; 
    const char* _ctrl  = "select id from ctrl order by id ;" ;           
    const char* _range = getenv("RANGE") ;

    if(argc > 1) _batch = argv[1] ; 
    if(argc > 2) _ctrl  = argv[2] ; 


    G4DAESensDet::MockupSD("DsPmtSensDet", new DybG4DAECollector ); // setup Geant4 SDs like NuWa/DetDesc does
    G4DAESensDet::MockupSD("DsRpcSensDet", new DybG4DAECollector );

    DsChromaRunAction_BeginOfRunAction(      // initializing G4DAEChroma, including hooking up trojan SD
         "G4DAECHROMA_CLIENT_CONFIG", 
         "G4DAECHROMA_CACHE_DIR", 
         "DsPmtSensDet" , 
         "G4DAECHROMA_DATABASE_PATH", 
          NULL, 
          "",
          true ); 

    G4DAEDatabase* db = G4DAEChroma::GetG4DAEChroma()->GetDatabase();

    std::vector<long> batch_id = getivec( db, _batch );
    std::vector<long> ctrl_id = getivec( db, _ctrl );

    int range[2] = {0} ;
    getintpair( _range, ':', range, range+1 ); 

    int seq = 0 ;
    for(size_t b=0 ; b < batch_id.size() ; ++b )
    {  
    for(size_t c=0 ; c < ctrl_id.size() ; ++c )
    {  
        int bid = batch_id[b];
        int cid = ctrl_id[c];
        printf("#propagate seq %3d bid %3d cid %3d \n", seq, bid, cid );
        if(!getenv("DRY")) Propagate(seq, bid, cid, range);   

        seq++ ;
    } 
    } 

    return 0 ; 
}




