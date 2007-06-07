

StringTest(){
  //load all external files
  gSystem->Load("$CMTPATH/DataStructure/MCEvent/i386-linux24/libMCEvent.so");

  //set initial variables
  const Int_t n = 4;
  Double_t ex[n];
  Double_t ey[n];
  Double_t x[n];
  Double_t y[n];
  TChain *chain[3];
  TH1F *h2[4];
  TH1F *h3[4][100];
  TH1F *h1 = new TH1F("h1","h1",500,0,1500);
  char filename[201][201][201];
    

  //loop over thicknesses
  for(int i=10; i<41; i=i+10)
    {  
      int runno = i/10-1;
      h2[runno] = new TH1F("h2","h2",500,0,1500);    

      //loop over all runs
      for(int j=1; j<50; j++)
	{
          h3[runno][j-1] = new TH1F("h3","h3",500,0,1500);

          //create string of the input filename
	  char result1[100];
	  char result2[100];
	  strcat(filename[i][j], "neutron_root/neutron_i");
	  sprintf(result1,"%d",i);
	  strcat(filename[i][j], result1 );
	  strcat(filename[i][j], result1 );
	  sprintf(result2,"%d",j);
	  strcat(filename[i][j], ".events_r");
	  strcat(filename[i][j], result2 );
	  strcat(filename[i][j], ".root");
	  cout<<i<<" and " <<j<<endl;
	  endl;
	 
	  //get correct tree
	  TFile *f = new TFile(filename[i][j],"read");
	  dywGLEvent* myevt = new dywGLEvent();	
	  TTree *tree = (TTree*) f->Get("event_tree");
	  tree->SetBranchAddress("dayabay_MC_event_output",&myevt);
	  
	  //create histogram h1 and fill with events_output data
	  for(int ii=0;ii<tree->GetEntries();ii++)
	    {
	      tree->GetEntry(ii);
	      //at this point, myevt has been filled with tree content for event ii  
	      int npe = myevt->hitSum_1;
	      if(npe < 600)
		{
		  if(npe > 400)
		    h3[runno][j-1]->Fill(npe);
		}		  
	    }
	  
	  //get total events
	  double events = h3[runno][j-1]->GetEntries();

	  //printouts
	  cout<<"Number of events: "<<events<<endl; 

	   
	  //add histogram to total
	  h2[runno]->Add(h3[runno][j-1]);

	  //Graph Portion
	  x[runno] = i;
	  y[runno] = events;
	  //ex[runno] = 0;
	  //ey[runno] = deviation;
	}
      //h3[runno]->Draw();
    }
  
  TGraph *g1 = new TGraph(n,x,y);
  TCanvas *c1 = new TCanvas("d1","Graph Draw Optioins",200,10,600,400);
  c1->SetGrid();
  g1->SetMarkerStyle(8);
  g1->SetMarkerColor(kBlue);
  g1->GetXaxis()->SetTitle("Inner AV Thickness");
  g1->GetYaxis()->SetTitle("Energy Resolution");
  g1->SetTitle("2MeV Gamma Energy Resolution");
  g1->Draw("AP");
  c1->SaveAs("resolution2set.eps");

  //Print Portion
  
return;
} 


