#include "CaptureMap.h"
#include "CaptureDB.h"

#include "TSystem.h"

#include "TFile.h"
#include "TTree.h"
#include "AbtEvent.h"


static const char* sample = "$ABERDEEN_HOME/DataModel/sample/run00027.root" ;


int capture_db(){
    /*
         Insert captured AbtEvent::Print outputs into SQLite DB 
    */

    TFile* f = TFile::Open(sample);
    TTree* t = (TTree*)f->Get("T") ;

    AbtEvent* evt = 0;
    t->SetBranchAddress( "trigger", &evt );
    Int_t n = (Int_t)t->GetEntries();

    CaptureDB db("try.db");
    
    int iok = 0;
    int ito = 0;
    
    for (Int_t i=0;i<n;i++) {
        t->GetEntry(i);
        
        string got ;
        {
            Capture c ;
            evt->Print();
            got = c.Gotcha();
        }
        
        /*
        // creation 
        stringstream ss ;
        ss << "insert into AbtEvent ('id','data') values (" << evt->GetSerialNumber() << ", '" << got << "' );" ;    
        db.Exec(ss.str().c_str());
        */
         
        // comparison 
        const char* expect = db.Get("AbtEvent", evt->GetSerialNumber() );
        const char* now    = got.c_str();
        
        ito += 1 ;
        if(strcmp(expect,now)==0){
            iok += 1;
        } else {
            cout << "ERROR mismatch " << got << endl ;
        }
    }
    cout << "capture_db " << iok << " match expectation out of " << ito << endl ;
    
}


int capture_db_read(){
    CaptureDB db("try.db");
    const char* dat = db.Get("AbtEvent", 1024) ;
    if(dat) cout << "selected ...  " << dat << endl ;
}



int main(){
    
    //capture_db_read();
    capture_db();
    
    
}



