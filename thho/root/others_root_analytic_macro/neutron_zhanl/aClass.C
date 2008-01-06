#define aClass_cxx
#include "aClass.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

void aClass::Loop()
{
//   In a ROOT session, you can do:
//      Root > .L aClass.C
//      Root > aClass t
//      Root > t.GetEntry(12); // Fill t data members with entry number 12
//      Root > t.Show();       // Show values of entry 12
//      Root > t.Show(16);     // Read and show values of entry 16
//      Root > t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
   fChain->SetBranchStatus("*",0);  // disable all branches
   fChain->SetBranchStatus("scint_1.*",1);  // activate branchname
   fChain->SetBranchStatus("vertex.*",1);  // activate branchname

// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
   if (fChain == 0) return;
   TFile* file = new TFile("file.root", "recreate");
   TCanvas* c1 = new TCanvas("c1", "", 600, 500);
  
   TH1F* energy = new TH1F("energy", "energy deposit", 100, 0, 10);
   //TH1F* eff_R = new TH1F("eff_R", "eff distribution vs R", 16,0, 2.56);
   TH1F* countR = new TH1F("countR", "counts in each sub-R",
		            16,0, 2.56);
   TH1F* countCutR = new TH1F("countCutR","counts above cut in each sub-R",
		               16, 0, 2.56);
   TH1F* countZ = new TH1F("countZ", "counts in each sub-Z",
		            16, -1.6, 1.6);
   TH1F* countCutZ = new TH1F("countCutZ","counts above cut in each sub-Z",
		               16, -1.6, 1.6);
   TH2F* count = new TH2F("count", "counts in each sub-volume",
		          4, 0, 2.56, 8, -1.6, 1.6 );
   TH2F* countCut = new TH2F("countCut", "counts above cut in each sub-volume",
		          4, 0, 2.56, 8, -1.6, 1.6 );

   TH2F* eff = new TH2F("eff", "efficiency in each sub-volume",
		          4, 0, 2.56, 8, -1.6, 1.6 );
   Float_t Rsquare[16];
   Float_t Zpoint[16];
   Float_t eff_R[16];
   Float_t eff_Z[16];
   Float_t eff_Rerror[16];
   Float_t eff_Zerror[16];

   /*Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;
   }*/
   Long64_t cutEvents = 0;
   Long64_t nentries = fChain->GetEntries();
   Long64_t Nentries = Nchain->GetEntries();
   Int_t deltaN = nentries - Nentries;
   for(Long64_t ievent=0; ievent<Nentries; ievent++)
   {
     //Note that the i-th event in neutron capture tree is not the i-th event
     //in event_tree. Here is a trick to get the right event.
     Nchain->GetEntry(ievent);
     fChain->GetEntry(eventid + 10000*Int_t((ievent + deltaN - eventid)/10000));
     
     if(capTarget == 1)//captrue on Gd
     {
       energy->Fill(scint_1_sE1+scint_1_sE2);
       if(fabs(vertex_z0[2]/1000)<0.8) //z cut,ignore z edge effect.
       {
         countR->Fill(pow(vertex_x0[2]/1000,2)+pow(vertex_y0[2]/1000,2));
         if(scint_1_sE1+scint_1_sE2>6)
	   countCutR->Fill(pow(vertex_x0[2]/1000,2)+pow(vertex_y0[2]/1000,2));
       }
       /*if(pow(vertex_x0[2]/1000,2)+pow(vertex_y0[2]/1000,2)<0.64)
       {
	 countZ->Fill(vertex_z0[2]/1000);
	 if(scint_1_sE1+scint_1_sE2>6)
           countCutZ->Fill(vertex_z0[2]/1000);
       }*/
       if(pow(xCap/1000,2)+pow(yCap/1000,2)<0.64)
       {
	 countZ->Fill(zCap/1000);
	 if(scint_1_sE1+scint_1_sE2>6)
           countCutZ->Fill(zCap/1000);
       }
       count->Fill(pow(vertex_x0[2]/1000,2)+pow(vertex_y0[2]/1000,2),
		      vertex_z0[2]/1000);
       if(scint_1_sE1+scint_1_sE2>6)
	 countCut->Fill(pow(vertex_x0[2]/1000,2)+pow(vertex_y0[2]/1000,2),
		      vertex_z0[2]/1000);
     }
   }
   for(Int_t ii=0; ii<16; ii++) 
   {
      Rsquare[ii] = 0.08 + ii*0.16;
      Float_t efficiency = (Float_t)countCutR->GetBinContent(ii+1)/
	                   countR->GetBinContent(ii+1);
      eff_R[ii] = efficiency;
      eff_Rerror[ii] = sqrt(efficiency*(1-efficiency)/
		            countR->GetBinContent(ii+1));
   }
   for(Int_t ii=0; ii<16; ii++) 
   {
      Zpoint[ii] = -1.5 + ii*0.2;
      Float_t efficiency = (Float_t)countCutZ->GetBinContent(ii+1)/
	                   countZ->GetBinContent(ii+1);
      eff_Z[ii] = efficiency;
      eff_Zerror[ii] = sqrt(efficiency*(1-efficiency)/
		            countZ->GetBinContent(ii+1));
   }
   for(Int_t i=0; i<8; i++)
   {
     for(Int_t j=0; j<4; j++)
     {
       Float_t efficiency = countCut->GetBinContent(j+1, i+1)/
	                   count->GetBinContent(j+1,i+1);
       efficiency = ((Int_t)(efficiency/0.001+0.5))/1000.0;
       //if((j==0&&i==0)||(j==0&&i==7))
         // efficiency = (Float_t)((Int_t)(efficiency/0.001)+5)/1000;
       eff->SetBinContent(j+1, i+1, efficiency);
     }
   }
   
   file->Write();
   cout<<"efficiency = "<<countCut->GetEntries()/count->GetEntries()<<endl;
   //efficiency = 0.918
   cout<<"nevents = "<<count->GetEntries()<<endl;
   // nevents = 127618
   eff->GetXaxis()->SetTitle("R^{2}(m^{2})");
   eff->GetXaxis()->CenterTitle();
   eff->GetYaxis()->SetTitle("Z(m)");
   eff->GetYaxis()->CenterTitle();
   eff->SetMarkerSize(1.6);
   eff->Draw("text");
   for(i=0; i<7; i++)
   {
     TLine* line = new TLine(0, -1.2+0.4*i, 2.56, -1.2+0.4*i);
     line->SetLineStyle(2);
     line->Draw();
   }
   for(j=0; j<3; j++)
   {
     TLine* line = new TLine(0.64+0.64*j, -1.6, 0.64+0.64*j, 1.6);	     
     line->SetLineStyle(2);
     line->Draw();
   }
   //energy->Draw();
   TCanvas* c2 = new TCanvas("c2","", 600,500);
   c2->cd();
   c2->SetGrid();
   TGraphErrors* eff_Rgr = new TGraphErrors(16, Rsquare, eff_R, 0, eff_Rerror);
   eff_Rgr->GetXaxis()->SetTitle("R^{2}(m^{2})");
   eff_Rgr->GetYaxis()->SetTitle("Efficiency");
   eff_Rgr->GetYaxis()->SetRangeUser(0.8,1.0);
   eff_Rgr->GetXaxis()->CenterTitle();
   eff_Rgr->GetYaxis()->SetTitleOffset(1.3);
   eff_Rgr->GetYaxis()->CenterTitle();
   eff_Rgr->SetMarkerSize(1.0);
   eff_Rgr->SetMarkerStyle(21);
   eff_Rgr->Draw("ALP");
   
   TCanvas* c3 = new TCanvas("c3","", 600,500);
   c3->cd();
   c3->SetGrid();
   TGraphErrors* eff_Zgr = new TGraphErrors(16, Zpoint, eff_Z, 0, eff_Zerror);
   eff_Zgr->GetXaxis()->SetTitle("Z(m)");
   eff_Zgr->GetYaxis()->SetTitle("Efficiency");
   eff_Zgr->GetYaxis()->SetRangeUser(0.8,1.0);
   eff_Zgr->GetXaxis()->CenterTitle();
   eff_Zgr->GetYaxis()->SetTitleOffset(1.3);
   eff_Zgr->GetYaxis()->CenterTitle();
   eff_Zgr->SetMarkerSize(1.0);
   eff_Zgr->SetMarkerStyle(21);
   eff_Zgr->Draw("ALP");
}
