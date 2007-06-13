

loop(){
   
   //
   //  demo of looping over all root files in a directory looking for an
   //  object called event_tree and doing a scan when found 
   //
   
   TString name("event_tree");
   FILE* pipe = gSystem->OpenPipe("ls *.root" , "r" );
   
   TString path ;
   TFile* f ;
   TTree* t ;
   
   while( path.Gets(pipe) ){
      cout << path << endl;
      f = TFile::Open( path );
	  t = (TTree*)gROOT->FindObject(name);
	  if( t == NULL ){
         cout << " no object called " << name << " in " << path << endl ; 
	  } else {
		 cout << " found " << name << " in " << path << endl ;
		 t->Scan("","","",10);
	  }
	  f->Close();
   }
   
   gSystem->Exit( gSystem->ClosePipe( pipe ));
}

