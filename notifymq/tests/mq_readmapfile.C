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

   const char* brn = "trigger" ;

   TTree* tree  = 0;
   AbtEvent* evt = 0 ;
   tree = (TTree*)mfile->Get("T");
   tree->SetBranchAddress( brn , &evt );

   //tree->GetEntry(0);  
   //Int_t entries = 0 ;


   Int_t entries = tree->GetEntries();
   for(Int_t i = 0 ; i < entries ; i++ ){
        tree->GetEntry(i);
        cout << endl << " entry " << i << endl ; 
        evt->Print();
   }

   while (1) {
      // this deletes the old tree and gets new one via copying from shared mem
      // ... highly-inefficient for large trees so use TTree::SetCircular(1000) on the producer side
      // to prevent the trees getting too big    
      tree = (TTree*)mfile->Get("T", tree);
      tree->SetBranchAddress( brn , &evt );  // its a new copy of the tree... so must set the branch addresses again

      Int_t n = tree->GetEntries();

      // hmm the circular TTree will trim one get too big 
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
