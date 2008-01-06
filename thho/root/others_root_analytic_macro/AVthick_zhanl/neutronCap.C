void neutronCap(Int_t thick)
{
  gInterpreter->ExecuteMacro("/dybdata2/zhanl/daya_style.C");
  TString filename;
  cout<<"thick:"<<thick<<endl;
  filename += "file"; 
  filename += thick ;
  filename +=  "mm.root";
  TFile* file = TFile::Open(filename);
  
  Float_t Hpeak, Hwidth, Gdpeak, Gdwidth;
 
  TCanvas* c1 = new TCanvas("","", 500, 400);
  neutronCapPE->Fit("gaus", "", "", 200, 300);
  Hpeak = gaus->GetParameter(1);
  Hwidth = gaus->GetParameter(2);
 
  TCanvas* c2 = new TCanvas("","", 500, 400);
  neutronCapPE->Fit("gaus", "", "", 850, 1000);
  Gdpeak = gaus->GetParameter(1);
  Gdwidth = gaus->GetParameter(2);

  cout<<"Hpeak = "<<Hpeak<<endl;
  cout<<"Hwidth = "<<Hwidth<<endl;
  cout<<"Gdpeak = "<<Gdpeak<<endl;
  cout<<"Gdwidth = "<<Gdwidth<<endl;

  Float_t neutronA, neutronB;
  const Float_t Henergy = 2.22;
  const Float_t Gdenergy = 8.0;
  
  neutronA = (Gdpeak-Hpeak)/(Gdenergy-Henergy);
  neutronB = Hpeak - Henergy*neutronA;
  
  cout<<"neutronA = "<<neutronA<<endl;
  cout<<"neutronB = "<<neutronB<<endl;
}
