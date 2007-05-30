
{

 // root:TColor
 gROOT->ProcessLine( Form(".x %s/DataStructure/MCEvent/cmt/load.C", gSystem->Getenv("DYW") ));


  TFile* f = new TFile("10.root","read");
  TTree* event_tree = (TTree*)f->Get("event_tree");
  TTree* trajectory_tree = (TTree*)f->Get("trajectory_tree");
  event_tree->AddFriend("trajectory_tree");

  // hmm how to do this with friends ???

  //TBranch* branch = trajectory_tree->GetBranch( "mc_trajectories");
  TBranch* branch = event_tree->GetBranch( "mc_trajectories");

  dywTrajectories* trajs = new dywTrajectories();
  branch->SetAddress( &trajs );
  

  Int_t nevent = event_tree->GetEntries();
  cout << " nevent " << nevent << endl ;

  TString ax1="x" ;
  TString ax2="y" ;

  TCanvas* c ;

  for( Int_t ievent=0 ; ievent < nevent ; ++ievent ){  
 
     c = new TCanvas ;
 
     trajectory_tree->Draw(Form("traj.point.%s:traj.point.%s",ax1.Data(),ax2.Data()),"") ;
     branch->GetEntry(ievent);
	 
	 vector<dywTrajectory> vt = trajs->traj ;
	 for (Int_t it=0;it<vt.size() ;it++) {
		 
        TString name = vt[it].particleName ;
	//    cout << it << " " << name << endl ;

        Color_t col ;
        if( name == "e+" ){ 
             col = kRed ;
	    } else if ( name == "neutron" ){
             col = kBlue ;
	    } else if ( name == "gamma" ){
             col = kCyan ;
	    } else if ( name == "deuteron" ){
             col = kBlue ;
	    } else if ( name == "proton" ){
             col = kGreen ;
	    } else if ( name == "C12[0.0]" ){
             col = kBlue ;
	    } else {
             col = kBlack ;
		}
			
        TString expr=Form("traj[%d].point.%s:traj[%d].point.%s",it,ax1.Data(),it,ax2.Data()); 
		trajectory_tree->SetLineColor(col);    
	    trajectory_tree->Draw(expr,"","LSAME") ;
        
     }
  }

}
