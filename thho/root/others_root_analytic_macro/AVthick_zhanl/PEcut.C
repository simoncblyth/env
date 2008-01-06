{
  gInterpreter->ExecuteMacro("/dybdata2/zhanl/daya_style.C");
  TCanvas* cc = new TCanvas("cc","", 600, 500);
  TLegend* lg = new TLegend(0.6, 0.8, 0.99, 0.99);
  
  TFile::Open("file10mm.root");
  neutronPE->GetXaxis()->SetRangeUser(650, 720);
  neutronPE->GetXaxis()->SetTitle("Photoelectron");
  neutronPE->GetYaxis()->SetRangeUser(0, 20);
  neutronPE->GetYaxis()->SetTitle("Events");
  //neutronPE->SetMarkerStyle(24);
  neutronPE->Draw();
  lg->AddEntry(neutronPE, "10 mm","L");

  TFile::Open("file15mm.root");
  //neutronPE->SetMarkerStyle(25);
  neutronPE->SetLineColor(kBlue);
  neutronPE->Draw("same");
  lg->AddEntry(neutronPE, "15 mm","L");
  
  TFile::Open("file20mm.root");
  //neutronPE->SetMarkerStyle(26);
  neutronPE->SetLineColor(kRed);
  neutronPE->Draw("same");
  lg->AddEntry(neutronPE, "20 mm","L");
  
  TFile::Open("file25mm.root");
  //neutronPE->SetMarkerStyle(27);
  neutronPE->SetLineColor(kGreen);
  neutronPE->Draw("same");
  lg->AddEntry(neutronPE, "25 mm","L");
  
  //TFile::Open("file30mm.root");
  //neutronPE->SetMarkerStyle(28);
  //neutronPE->SetLineColor(kYellow);
  //neutronPE->Rebin(2);
  //neutronPE->Draw("same");
  //lg->AddEntry(neutronPE, "30 mm","L");

  lg->SetBorderSize(0);
  lg->Draw();
}
