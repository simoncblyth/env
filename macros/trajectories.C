
{

 //
 // gROOT->ProcessLine( Form(".x %s/DataStructure/MCEvent/cmt/load.C", gSystem->Getenv("DYW") ))
 //


  TFile* f = new TFile("dummy.root","read");
  TTree* traj_t = (TTree*)f->Get("trajectory_tree");
  TBranch* branch = traj_t->GetBranch( "mc_trajectories");
  
  dywTrajectories* trajs = new dywTrajectories();
  branch->SetAddress( &trajs );

  Int_t nevent = traj_t->GetEntries();
  cout << " nevent " << nevent << endl ;

  TString ax1="x" ;
  TString ax2="y" ;
  trajectory_tree->Draw(Form("traj.point.%s:traj.point.%s",ax1,ax2),"") ;

  for( Int_t ievent=0 ; ievent < nevent ; ++ievent ){  
  
     branch->GetEntry(ievent);
	 
	 vector<dywTrajectory> vt = trajs->traj ;
	 for (Int_t it=0;it<vt.size() ;it++) {
		 
        TString name = vt[it].particleName ;
	    cout << it << " " << name << endl ;

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
			
        TString expr=Form("traj[%d].point.%s:traj[%d].point.%s",it,ax1,it,ax2); 
		trajectory_tree->SetLineColor(col);    
	    trajectory_tree->Draw(expr,"","LSAME") ;
        
     }
  }

}
