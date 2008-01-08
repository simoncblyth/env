//
//   Simple demo code for propagating PMT positions and ids from a PMTPositionTree 
//   into an array of PMT structs, allowing fast access for subsequent
//   plotting
//
//   Cint interactive test with : 
//
//       .L pmt_positions.C
//       pmt_positions_test( "../macros/test-optical.root" )
//
//

typedef struct { Int_t id ; Double_t x,y,z ; } PMT ;


// NB this function is named the same as this file without the extension 
PMT* pmt_positions( const char* path = "../macros/test-optical.root" , Int_t nmax = 192 ){
    
    PMT* pmt = new PMT[nmax] ;
    pmt_positions_load( path , nmax , pmt );
    //pmt_positions_dump( nmax , pmt );

    return pmt ;
}

Int_t pmt_positions_load( const char* path , Int_t nmax , PMT* pmt  ){ 

  TFile f(path);  
  TTree* ppt = (TTree*)f.Get("PMTPositionTree") ;
  //TTree* ppt = (TTree*)gROOT->FindObject("PMTPositionTree") ;

  Int_t nent = Int_t(ppt->GetEntries()) ;
  if ( nent > nmax || nent < 0 ) return -1 ;

  static PMT pj ;

  ppt->SetBranchAddress("iPMT", &pj.id );
  ppt->SetBranchAddress("xPMT", &pj.x );
  ppt->SetBranchAddress("yPMT", &pj.y );
  ppt->SetBranchAddress("zPMT", &pj.z );
  
  for( Int_t j=0 ; j < nent ; j++ ){
     ppt->GetEntry(j); 
     pmt[j].id = pj.id ;
     pmt[j].x  = pj.x ;
     pmt[j].y  = pj.y ;
     pmt[j].z  = pj.z ;
  }
  return 0 ;
}

void pmt_positions_dump( Int_t nmax , PMT* pmt ){
   static PMT pj ;
   for( Int_t j = 0 ; j < nmax ; j++ ){
       pj = pmt[j] ;
       cout << Form(" %d %f %f %f " , pj.id , pj.x , pj.y , pj.z ) << endl ;
   }
}

void pmt_positions_test( const char* path ){
  
  static const Int_t nmax = 192 ;
  PMT pmt[nmax] ; 
  pmt_positions_load( path , nmax , pmt );
  pmt_positions_dump( nmax , pmt );
  
}
