#include <TFile.h>
#include <TGraphErrors.h>
#include <TH1F.h>

void Parameter(Int_t thick )
{
  gInterpreter->ExecuteMacro("/dybdata2/zhanl/daya_style.C");
  TString prename = "gamma";
  prename += thick;
  TString filename;
  Float_t energy[ ] = {1.0,  2.0,  4.0,  6.0,  8.0};
  Int_t Eindex[] = {1, 2, 4, 6, 8};
  Float_t peak[5];
  Float_t width[5];
  Float_t resolution[5];
	    
  for(Int_t i=0; i<5; i++)
  {
    filename = prename + "mm" + Eindex[i] + "MeV.root";
    TFile::Open(filename);
    TH1F* pe = new TH1F("pe", "", 180, 20*Eindex[i], 200*Eindex[i]);
    event_tree->Draw("hitSum_1>>pe");
    peak[i] = pe->GetMean();
    width[i] = pe->GetRMS();
    pe->Fit("gaus","Q","", peak[i]-3*width[i], peak[i]+3*width[i]);
    peak[i] = gaus->GetParameter(1);
    width[i] = gaus->GetParameter(2);
    pe->Fit("gaus","Q","", peak[i]-3*width[i], peak[i]+3*width[i]);
    peak[i] = gaus->GetParameter(1);
    width[i] = gaus->GetParameter(2)/sqrt(5000);
    resolution[i] = 100*gaus->GetParameter(2)/peak[i];
  }
  
  TCanvas* c1 = new TCanvas("c1", "", 600,500);
  TGraphErrors* gr1 = new TGraphErrors(5, energy, peak, 0, width);
  gr1->Draw("AP");
  gr1->Fit("pol1", "Q", "", 1, 8);
  cout<<"positronA = "<<pol1->GetParameter(1)<<endl;
  cout<<"positronB = "<<pol1->GetParameter(0)<<endl;
  
  TF1 *func = new TF1("fitf", "[0]/sqrt(x)", 0, 10);
  func->SetParameter(0, 12);
  TCanvas* c2 = new TCanvas("c2", "", 600,500);
  TGraph* graph = new TGraph(5, energy, resolution);
  graph->Draw("AP");
  graph->GetXaxis()->SetTitle ("Energy(MeV)");
  graph->GetXaxis()->CenterTitle();
  graph->GetXaxis()->SetRangeUser(0,10);
  graph->GetYaxis()->SetTitle ("Resolution (%)");
  graph->GetYaxis()->CenterTitle();
  graph->GetYaxis()->SetRangeUser(0,15);
  graph->Fit("fitf", "", "", 1, 8);
 
}

