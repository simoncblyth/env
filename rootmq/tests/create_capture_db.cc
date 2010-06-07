#include <sys/stat.h>	

#include "Capture.h"
#include "CaptureDB.h"

#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"

#include "AbtEvent.h"

static const char* sample = "$ABERDEEN_HOME/DataModel/sample/run00027.root" ;

int create_capture_db(){
 
    TFile* f = TFile::Open(sample);
    TTree* t = (TTree*)f->Get("T") ;

    const char* path = gSystem->ExpandPathName(Form("%s.db",sample));
    
    struct stat s_buf;
    int rc = stat(path,&s_buf) ;
    if( -1 == rc ){
        cout << "create_capture_db : DB does not exist : " << path << endl ;
    } else if ( 0 == rc ){
        cout << "create_capture_db : DB exists already : " << path << " ... delete it and re-run to recreate " << endl ;
        exit(0);
    }
    
    CaptureDB db(path);
    db.Exec("create table AbtEvent (id INTEGER PRIMARY KEY,data TEXT);");
    db.Exec("create table AbtRunInfo (id INTEGER PRIMARY KEY,data TEXT);");
    
    AbtEvent* evt = 0;
    t->SetBranchAddress( "trigger", &evt );
    Int_t n = (Int_t)t->GetEntries();
    
    for (Int_t i=0;i<n;i++) {
        t->GetEntry(i);
        string found ;
        {
            Capture c ;
            evt->Print();
            found = c.Gotcha();
        }
        stringstream ss ;
        ss << "insert into AbtEvent ('id','data') values (" << evt->GetSerialNumber() << ", '" << found << "' );" ;    
        db.Exec(ss.str().c_str());           
    } 
    return EXIT_SUCCESS ;
}

int main()
{
    return create_capture_db();
}


