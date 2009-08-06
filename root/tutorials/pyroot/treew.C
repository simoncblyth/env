void treew() {
     TFile f("test.root","recreate");
     TNtuple *ntuple = new TNtuple("ntuple","Demo","px:py:pz:random:i");
     Float_t px, py, pz;
     for ( Int_t i=0; i<10000000; i++) {
        gRandom->Rannor(px,py);
        pz = px*px + py*py;
        Float_t random = gRandom->Rndm(1);
        ntuple->Fill(px,py,pz,random,i);
        if (i%1000 == 1) ntuple->AutoSave("SaveSelf");
     }
   }

