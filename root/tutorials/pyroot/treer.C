void treer() {
      TFile f("test.root");
      TTree *ntuple = (TTree*)f.Get("ntuple");
      TCanvas c1;
      Int_t first = 0;
      while(1) {
         if (first == 0) ntuple->Draw("px>>hpx", "","",10000000,first);
         else            ntuple->Draw("px>>+hpx","","",10000000,first);
         first = (Int_t)ntuple->GetEntries();
         c1.Update();
         gSystem->Sleep(1000); //sleep 1 second
         ntuple->Refresh();
      }
   }

