
{

 //
 // gROOT->ProcessLine( Form(".x %s/DataStructure/MCEvent/cmt/load.C", gSystem->Getenv("DYW") ))
 //


  TFile* f = new TFile("dummy.root","read");
  dywTrajectories* trajs = new dywTrajectories();
  TTree* traj_t = (TTree*)f->Get("trajectory_tree");

  //tree->SetBranchAddress("dayabay_MC_track_output",&trk);  
  TBranch* branch = traj_t->GetBranch( "mc_trajectories");
  branch->SetAddress( &trajs );

  Int_t nevent = traj_t->GetEntries();
  cout << " nevent " << nevent << endl ;

  for( Int_t ievent=0 ; ievent < nevent ; ++ievent ){  
     branch->GetEntry(ievent);
	 vector<dywTrajectory> vt = trajs->traj ;
	 for (Int_t it=0;it<vt.size() ;it++) {
	    cout << it << " " << vt[it].particleName << endl ;
     }
  }

}
