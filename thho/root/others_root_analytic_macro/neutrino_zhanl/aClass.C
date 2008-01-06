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
    fChain->SetBranchStatus("pmtHitData_1.*",1);  // activate branchname
    fChain->SetBranchStatus("nHit",1);  // activate branchname
    fChain->SetBranchStatus("hitSum_1*",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
   if (fChain == 0) return;
   TFile* file = new TFile("fileNoAcrylic.root", "recreate");
       
   TH1F* drift = new TH1F("drift", "distance of neutron drift", 50, 0, 50);
   TH1F* driftx = new TH1F("driftx",
		   "distance of neutron drift in x direction", 60, -30, 30);
   TH1F* driftz = new TH1F("driftz",
                   "distance of neutron drift in z direction", 60, -30, 30);
   TH1F* drift_LS = new TH1F("drift_LS",
                   "distance of neutron drift in LS", 50, 0, 50);
   TH1F* drift_GdLS = new TH1F("drift_GdLS",
                   "distance of neutron drift in GdLS", 100, 0, 50);
   TH1F* drift_LSx = new TH1F("drift_LSx",
                 "distance of neutron drift in x direction in LS", 80, -40, 40);
   TH1F* drift_GdLSx = new TH1F("drift_GdLSx",
               "distance of neutron drift in z direction in GdLS", 60, -30, 30);
   TH1F* drift_LSy = new TH1F("drift_LSy",
                 "distance of neutron drift in y direction in LS", 80, -40, 40);
   TH1F* drift_GdLSy = new TH1F("drift_GdLSy",
               "distance of neutron drift in y direction in GdLS",120, -30, 30);
   TH1F* drift_LSz = new TH1F("drift_LSz",
                 "distance of neutron drift in z direction in LS", 80, -40, 40);
   TH1F* drift_GdLSz = new TH1F("drift_GdLSz",
               "distance of neutron drift in z direction in GdLS", 120, -30,30);
   TH1F* drift_LSr = new TH1F("drift_LS_Radius",
             "distance of neutron drift in radius direction in LS", 80, 0, 40);
   TH1F* drift_GdLSr = new TH1F("drift_GdLS_Radius",
            "distance of neutron drift in radius direction in GdLS", 80,0,40);
   TH1F* spill_in = new TH1F("spill_in",
	              "distance of neuton spill into Gd-LS", 50, 0, 50);
   TH1F* spill_out = new TH1F("spill_out",
                      "distance of neuton spill out of Gd-LS", 50, 0, 50);
   TH1F* time_LS = new TH1F("time_LS", "capture time in LS", 100, 0, 1500);
   TH1F* time_GdLS = new TH1F("time_GdLS", "capture time in GdLS", 200, 0, 400);
   TH1F* riseTime = new TH1F("riseTime","rise edge of capture time ",100, 0, 3);
   TH1F* dropTime = new TH1F("dropTime",
                              "drop edge of capture time ", 200, 100, 300);
   TH1F* gammaN = new TH1F("gammaN","number of gamma",10, 0, 10);
   TH1F* gammaE = new TH1F("gammaE","gamma energy",100, 0, 10);
   TH1F* spectrum = new TH1F("spectrum", "p.e. number", 300, 0, 3000);
   
   TH1F* hitTime = new TH1F("hitTime", "photon time", 20, 0, 10);
   TH1F* pHitTime = new TH1F("pHitTime", "positron signal time", 50, 0, 1000);
   TH1F* nHitTime = new TH1F("nHitTime", "neutron signal time", 200, 0, 400);
   TH1F* GdVertex = new TH1F("GdVertex","neutron production vertex",62,0,4.96);
   TH1F* GdCapPos = new TH1F("GdCapPos", "neutron capture position", 62,0,4.96);
   TH1F* HCapPos = new TH1F("HCapPos", "neutron capture position", 62, 0, 4.96);
   
   TH1F* GdVertexcut = new TH1F("GdVertexcut",
		       "neutron production vertex with 6MeV cut" ,62,0,4.96);
   TH1F* GdCapPoscut = new TH1F("GdCapPoscut", 
		       "neutron capture position with 6MeV cut", 62,0,4.96);
  
   TH2F* pe_E = new TH2F("pe_E", "pe distribution vs E",200,0,20, 1200,0,2400);
   TH2F* positronPE_E = new TH2F("positronPE_E", "pe distribution vs E",
		                                100,0,20, 600,0,1200);
   TH2F* neutronPE_E = new TH2F("neutronPE_E", "pe distribution vs E",
		                                   100,0,20, 600,0,1200);
   TH1F* positronPE = new TH1F("positronPE","positron PE spectrum", 100,0,1200);
   TH1F* positronE = new TH1F("positronE","positron E spectrum", 
		              600,0,1200/113.5);
   TH1F* neutronPE = new TH1F("neutronPE","neutron PE spectrum", 100,0,1200);
   TH1F* neutronE = new TH1F("neutronE","neutron E spectrum", 
		              600,0,1200/113.5);
   TH2F* positronPE_R = new TH2F("positronPE_R",  "positron PE nubmer of unit 
		        energy vs. R passing 6MeV cut", 100, 0, 4, 200, 0, 200);
   TH2F* positronPE_Z = new TH2F("positronPE_Z",  "positron PE nubmer of unit 
		        energy vs. Z passing 6MeV cut", 100, 0, 2, 200, 0, 200);
   TH2F* positronVertex = new TH2F("positronVertex", "positron Vertex of passing
		        6 MeV but not 1 MeV energy cut", 250, 0, 4, 200, 0, 2);
   TH2F* neutronVertex = new TH2F("neutronVertex", "neutron Vertex of not
		        passing 6 MeV  energy cut", 250, 0, 4, 200, 0, 2);

   //Float_t energyCutEff; 
   Float_t timeCutEff1;
   Float_t timeCutEff2;
   Float_t GdCapRatio;   
   Int_t nevents;
   Int_t HCap, GdCap, CCap;
   Int_t positronHit, neutronHit;

   /*Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;
   }*/
   
   Long64_t nentries = fChain->GetEntries();
   Long64_t Nentries = Nchain->GetEntries();
   Int_t deltaN = nentries - Nentries; 
   
   for(Long64_t ievent=0; ievent<Nentries; ievent++)
   {
     //Note that the i-th event in neutron capture tree is not the i-th event
     //in event_tree. Here is a trick to get the right event. Number of entries
     //in each tree is 5000.
     Nchain->GetEntry(ievent);
     fChain->GetEntry(eventid + 5000*Int_t((ievent + deltaN - eventid)/5000));
     if(capTarget==1
	//&&pow(vertex_x0[1]/1000,2)+pow(vertex_y0[1]/1000,2)<1
	//&&fabs(vertex_z0[1]/1000)<1
	&&scint_1_sE1>6+1.022+vertex_ke[1]
        &&scint_1_sE2<0.01 //no E dep in LS
	)
        pe_E->Fill(scint_1_sE1+scint_1_sE2-vertex_ke[2], hitSum_1);
	//ignore the proton energy deposit
   }
   pe_E->ProfileX();
   pe_E_pfx->Fit("pol1", "", "", 9, 15);

   for(Long64_t ievent=0; ievent<Nentries; ievent++)
   {
     //Note that the i-th event in neutron capture tree is not the i-th event
     //in event_tree. Here is a trick to get the right event.
     Nchain->GetEntry(ievent);
     fChain->GetEntry(eventid + 5000*Int_t((ievent + deltaN - eventid)/5000));
    
     /*for(Int_t iphoton = 0; iphoton < pmtHitData_1_; iphoton++)
     {
       //if(pmtHitData_1_t[iphoton]<10000)
       //  hitTime->Fill(pmtHitData_1_t[iphoton]/1000);
       //if(pmtHitData_1_t[iphoton]<1000)
       //  pHitTime->Fill(pmtHitData_1_t[iphoton]);
       if(capTarget == 1&&pmtHitData_1_t[iphoton]>1000)
         nHitTime->Fill(pmtHitData_1_t[iphoton]/1000);
     }*/
     positronHit = 0;
     neutronHit = 0;
     if(capTarget==1)
     {
       for(Int_t iphoton = 0; iphoton < pmtHitData_1_; iphoton++)
       {
         if(pmtHitData_1_t[iphoton]<500)// positron hit<500ns.
           positronHit++;
	 else
	   neutronHit++;
       }
       //if(ievent<10)
       //cout<<"positronHit = "<<positronHit<<"neutronHit = "<<neutronHit<<endl;
       //use parameters in 1cm acrylic case
       neutronE->Fill((neutronHit-pol1->GetParameter(0))/pol1->GetParameter(1));
       neutronPE->Fill(neutronHit); 
       if(neutronHit>6.0*pol1->GetParameter(1)+pol1->GetParameter(0))
       {
	 //passing 6 MeV cut
	 positronPE_E->Fill(vertex_ke[1]+1.022, positronHit);
	 neutronPE_E->Fill(scint_1_sE1+scint_1_sE2-vertex_ke[2]-vertex_ke[1]
				                         -1.022, neutronHit);
	 positronPE->Fill(positronHit);
	 //10.6 and 116.6 come from positronPE_E.
	 positronE->Fill((positronHit+10.6)/116.6);
	 
	 positronPE_R->Fill(pow(vertex_x0[1]/1000,2)+pow(vertex_y0[1]/1000,2),
			    positronHit/(1.022+vertex_ke[1]));
	 positronPE_Z->Fill(fabs(vertex_z0[1]/1000),
			    positronHit/(1.022+vertex_ke[1]));
	 //not passing 1 MeV cut
	 if(positronHit<1.0*116.6-10.6)
	   positronVertex->Fill(pow(vertex_x0[1]/1000,2)+pow(vertex_y0[1]/1000,
				    2), fabs(vertex_z0[1]/1000));
       }
       else
	   neutronVertex->Fill(pow(vertex_x0[2]/1000,2)+
		      pow(vertex_y0[2]/1000,2),fabs(vertex_z0[2]/1000));
     }
   
     Float_t distance = sqrt((xCap-vertex_x0[0])*(xCap-vertex_x0[0])/100
                              +(yCap-vertex_y0[0])*(yCap-vertex_y0[0])/100
                              +(zCap-vertex_z0[0])*(zCap-vertex_z0[0])/100);
     Float_t deltaR = sqrt(xCap*xCap+yCap*yCap)/10
                - sqrt(vertex_x0[0]*vertex_x0[0]+vertex_y0[0]*vertex_y0[0])/10;
					   
     drift->Fill(distance);
     driftx->Fill((xCap-vertex_x0[0])/10);
     driftz->Fill((zCap-vertex_z0[0])/10);
     
     if(capTarget==1)
     {
       gammaN->Fill(gammaNum);
       if(capTime<200)
         timeCutEff1 = timeCutEff1 +1;
       if(capTime<200&&capTime>1)
         timeCutEff2 = timeCutEff2 +1;
       
       if(capTime<3)
          riseTime->Fill(capTime);
       if(capTime<300&&capTime>100)
          dropTime->Fill(capTime);
       for(Int_t i=0; i<gammaNum; i++)
	  gammaE->Fill(capGammaE[i]);
       if(vertex_z0[0]*vertex_z0[0]<1590*1590)
       {
	 GdVertex->Fill(vertex_x0[0]/1000*vertex_x0[0]/1000
			 +vertex_y0[0]/1000*vertex_y0[0]/1000);
         GdCapPos->Fill(xCap/1000*xCap/1000+yCap/1000*yCap/1000);
         if(neutronHit>6.0*hitSum_1/(scint_1_sE1+scint_1_sE2))
	 {
            GdVertexcut->Fill(vertex_x0[0]/1000*vertex_x0[0]/1000
			 +vertex_y0[0]/1000*vertex_y0[0]/1000);
            GdCapPoscut->Fill(xCap/1000*xCap/1000+yCap/1000*yCap/1000);
	 }
       }
     }

     if(capTarget==2&&vertex_z0[0]*vertex_z0[0]<1590*1590)
       HCapPos->Fill(xCap/1000*xCap/1000+yCap/1000*yCap/1000);
      
     if((vertex_x0[0]*vertex_x0[0]+vertex_y0[0]*vertex_y0[0]>1610*1610
        ||vertex_z0[0]*vertex_z0[0]>1610*1610)
        &&xCap*xCap+yCap*yCap>1610*1610||zCap*zCap>1610*1610)
     {
       //drift in LS
       drift_LS->Fill(distance);
       time_LS->Fill(capTime);
       drift_LSx->Fill((xCap-vertex_x0[0])/10);
       drift_LSy->Fill((yCap-vertex_y0[0])/10);
       drift_LSz->Fill((zCap-vertex_z0[0])/10);
       if(deltaR<0)drift_LSr->Fill(-deltaR);
     }
                                              
     if((vertex_x0[0]*vertex_x0[0]+vertex_y0[0]*vertex_y0[0]<1600*1600
        &&vertex_z0[0]*vertex_z0[0]<1600*1600)
        &&xCap*xCap+yCap*yCap<1600*1600&&zCap*zCap<1600*1600)
     {
       //drift in GdLS
       drift_GdLS->Fill(distance);
       time_GdLS->Fill(capTime);
       drift_GdLSx->Fill((xCap-vertex_x0[0])/10);
       drift_GdLSy->Fill((yCap-vertex_y0[0])/10);
       drift_GdLSz->Fill((zCap-vertex_z0[0])/10);
       if(deltaR>0)drift_GdLSr->Fill(deltaR);
       
       /*if(capTime<200)
         timeCutEff1 = timeCutEff1 +1;
       if(capTime<200&&capTime>1)
         timeCutEff2 = timeCutEff2 +1;
       */

       nevents++; 
       if(capTarget == 1)
         GdCap++;
       if(capTarget == 2)
	 HCap++;
       if(capTarget == 3)
	 CCap++;
     }

     if((vertex_x0[0]*vertex_x0[0]+vertex_y0[0]*vertex_y0[0]>1610*1610
        ||vertex_z0[0]*vertex_z0[0]>1610*1610)
        &&xCap*xCap+yCap*yCap<1600*1600&&zCap*zCap<1600*1600)
     {
       //spill in
       spill_in->Fill(distance);
     }
         
     if(vertex_x0[0]*vertex_x0[0]+vertex_y0[0]*vertex_y0[0]<1600*1600
        &&vertex_z0[0]*vertex_z0[0]<1600*1600
        &&(xCap*xCap+yCap*yCap>1610*1610||zCap*zCap>1610*1610))
     {
       //spill out 
       spill_out->Fill(distance);
     }
   }
   
   cout<<"neutronEff = "<<positronPE_R->GetEntries()/
	        (positronPE_R->GetEntries()+neutronVertex->GetEntries())<<endl; 
   cout<<"positronEff = "<<(positronPE_R->GetEntries()-positronVertex->
		           GetEntries())/positronPE_R->GetEntries()<<endl; 
   cout<<"nevents = "<<nevents<<endl;
   cout<<"GdCap = "<<GdCap<<endl;
   cout<<"HCap = "<<HCap<<endl;
   cout<<"CCap = "<<CCap<<endl;
   cout<<"timeCutEff1 = "<<timeCutEff1/gammaN->GetEntries()<<endl; 
   cout<<"timeCutEff2 = "<<timeCutEff2/gammaN->GetEntries()<<endl; 
   cout<<" Gd capture ratio: "<<(Float_t)GdCap/nevents<<endl; 
   cout<<" H capture ratio: "<<(Float_t)HCap/nevents<<endl; 
   cout<<" C capture ratio: "<<(Float_t)CCap/nevents<<endl; 
   
   file->Write();
   /*drift_GdLS->GetXaxis()->SetTitle("Distance (cm)");
   drift_GdLS->GetXaxis()->CenterTitle();
   drift_GdLS->GetXaxis()->SetTitleSize(0.05);
   drift_GdLS->SetLineWidth(2);
   drift_GdLS->Draw(); 
   */
   
   //gammaE->Draw();
   //spectrum->Draw();
   //nHitTime->Draw();

   /* time_GdLS->GetXaxis()->SetTitle("Capture Time (#mus)");
    time_GdLS->GetXaxis()->CenterTitle();
    time_GdLS->GetXaxis()->SetTitleSize(0.05);
    time_GdLS->SetLineWidth(2);
    time_GdLS->Draw(); 
   */

   /* riseTime->GetXaxis()->SetTitle("Capture Time (#mus)");
    riseTime->GetXaxis()->CenterTitle();
    riseTime->GetXaxis()->SetTitleSize(0.05);
    riseTime->SetLineWidth(2);
    riseTime->Draw(); 
   */
    
    /*TCanvas* c2 = new TCanvas("c2","", 600, 500 );
    c2->cd();
    GdVertexcut->GetXaxis()->SetTitle("R^{2}(m^{2})");
    GdVertexcut->GetXaxis()->CenterTitle();
    GdVertexcut->GetXaxis()->SetTitleSize(0.05);
    GdVertexcut->SetLineWidth(2);
    GdVertexcut->Draw();
    TLine* line1 = new TLine(2.56,0,2.56,950 );
    line1->SetLineWidth(2);
    line1->SetLineColor(kBlue);
    line1->Draw(); 
    */

    /*TCanvas* c2 = new TCanvas("c2","", 600, 500 );
    c2->cd();
    GdCapPos->GetXaxis()->SetTitle("R^{2}(m^{2})");
    GdCapPos->GetXaxis()->CenterTitle();
    GdCapPos->GetXaxis()->SetTitleSize(0.05);
    GdCapPos->SetLineWidth(2);
    GdCapPos->SetLineStyle(1);
    GdCapPos->Draw();
    HCapPos->SetLineWidth(2);
    HCapPos->SetLineStyle(2);
    HCapPos->Draw("same");
    TLegend *legend = new TLegend(0.35,0.65,0.50,0.77);
    legend->AddEntry(GdCapPos,"Gd capture vertex","L");
    legend->AddEntry(HCapPos,"H capture vertex","L");
    legend->Draw(); 
    */
    /*TCanvas* c3 = new TCanvas("c3","", 600, 500 );
    c3->cd();
    GdCapPoscut->GetXaxis()->SetTitle("R^{2}(m^{2})");
    GdCapPoscut->GetXaxis()->CenterTitle();
    GdCapPoscut->GetXaxis()->SetTitleSize(0.05);
    GdCapPoscut->SetLineWidth(2);
    GdCapPoscut->SetLineStyle(1);
    GdCapPoscut->Draw();
    HCapPos->SetLineWidth(2);
    HCapPos->SetLineStyle(2);
    HCapPos->Draw("same");
    TLegend *legend = new TLegend(0.35,0.65,0.50,0.77);
    legend->AddEntry(GdCapPoscut,"Gd capture vertex","L");
    legend->AddEntry(HCapPos,"H capture vertex","L");
    legend->Draw(); 
    */
    //spill_in->Draw();
   //spill_out->Draw();
    TCanvas* c21 = new TCanvas("c21","", 600, 500 );
    c21->cd();
    positronPE_R->Draw();
    TCanvas* c22 = new TCanvas("c22","", 600, 500 );
    c22->cd();
    positronPE_Z->Draw();
    TCanvas* c23 = new TCanvas("c23","", 600, 500 );
    c23->cd();
    positronVertex->Draw();
    TCanvas* c24 = new TCanvas("c24","", 600, 500 );
    c24->cd();
    neutronVertex->Draw();

}
