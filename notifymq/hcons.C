{
   // based on $ROOTSYS/tutorials/net/hcons.C
   //   rootn.exe -l hcons.C  
   //
   //   getting dupes... when feed in new events via MQ 

 
   gROOT->Reset();
   gSystem->Load("$ABERDEEN_HOME/DataModel/lib/libAbtDataModel.so");

   mfile = TMapFile::Create("mq_mapfile.map");
   mfile->Print();
   mfile->ls();

   TTree* tree  = 0;
   AbtEvent* evt = 0 ;
   tree = (TTree*)mfile->Get("T");
   tree->SetBranchAddress( "trigger" , &evt );

   //tree->GetEntry(0);  
   //Int_t entries = 0 ;


   Int_t entries = tree->GetEntries();
   for(Int_t i = 0 ; i < entries ; i++ ){
        tree->GetEntry(i);
        cout << endl << " entry " << i << endl ; 
        evt->Print();
   }

   while (1) {
      tree = (TTree*)mfile->Get("T", tree);
      Int_t n = tree->GetEntries();
      if( n > entries ){
          cout << "received additional entries " << n << " " << entries << endl ;
          for(Int_t i = TMath::Max(entries - 1,0) ; i < n ; i++ ){
              tree->GetEntry(i);
              cout << endl << " entry " << i << endl ; 
              evt->Print();
          }  
          entries = n ;
      }
      gSystem->Sleep(1000);   // sleep for 1 seconds
      if (gSystem->ProcessEvents()) break;
   }
   cout << "Done " << endl ;
}
