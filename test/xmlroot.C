{
   TFile *f = TFile::Open("Example.xml","recreate");
   TH1F *h = new TH1F("h","test",1000,-2,2);
   h->FillRandom("gaus");
   h->Write();
   delete f;
}
