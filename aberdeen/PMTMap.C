//
//   Simple demo code for loading PMT positions and ids from a PMTPositionTree 
//   into a PMTMap class that contains a pointer to an array of PMT structs, 
//   allowing fast/convenient access for subsequent manipulations/plotting
//
//   Also demonstrates cint interactive development of a class, 
//   see   ftp://root.cern.ch/root/doc/7CINT.pdf 
//
//   Cint interactive test with : 
//
//       .L PMTMap.C
//       PMTMap* pm = new PMTMap ; pm->Load("path/to/g4dyb/rootfile/for/geometry/of/interest.root") ; pm->Dump()
//
//   Should result in smth like :
//
//       PMTMap::Load loading from path,treename : ../macros/test-optical.root,PMTPositionTree found entries: 192
//    
//    0 2328.000000 0.000000 -1806.875000 
//    1 2248.675324 602.530737 -1806.875000 
//    2 2016.107140 1164.000000 -1806.875000 
//    3 1646.144587 1646.144587 -1806.875000 
//    4 1164.000000 2016.107140 -1806.875000 
//
//   Use the loaded map ...
//        pm.fPMT[1].y
//
//#include <iostream.h>

typedef struct { Int_t id ; Double_t x,y,z ; } PMT ;

class PMTMap {

public:
    Int_t fEntries ;
    PMT* fPMT ;

    PMTMap();
    ~PMTMap();
    
    void Load( const char* path = "../macros/test-optical.root" ,  const char* name = "PMTPositionTree" , Int_t nmax=1000  );
    void Dump();

    // accessors to be added :
    //Double_t GetX( Int_t id );
    //Double_t GetY( Int_t id );
    //Double_t GetZ( Int_t id );
    //Double_t GetTheta( Int_t id );
    //Double_t GetPhi( Int_t id );


    static void Test(const char* path);
};


// Double_t PMTMap::GetX( Int_t id ){
//}


void PMTMap::PMTMap(){
  fPMT = NULL ;
  fEntries = -1 ;
}

void PMTMap::Load( const char* path , const char* treename , Int_t nmax ){

  TFile f(path);  
  TTree* ppt = (TTree*)f.Get( treename ) ;
  Int_t nent = Int_t(ppt->GetEntries()) ;
  
  cout << "PMTMap::Load loading from path,treename :  " << path << "," << treename << " found entries: " << nent << endl ;
  
  if ( nent > nmax || nent <= 0 ) {
     cout << "PMTMap::Load FATAL ERROR path,treename : " << path << "," << treename << " did not yield expected entries : " << nent << endl ;
     assert(0);
  }

  fEntries = nent ;
  fPMT = new PMT[nent] ;

  static PMT pj ;
  ppt->SetBranchAddress("iPMT", &pj.id );
  ppt->SetBranchAddress("xPMT", &pj.x );
  ppt->SetBranchAddress("yPMT", &pj.y );
  ppt->SetBranchAddress("zPMT", &pj.z );
 
  for( Int_t j=0 ; j < nent ; j++ ){
     ppt->GetEntry(j); 
     fPMT[j].id = pj.id ;
     fPMT[j].x  = pj.x ;
     fPMT[j].y  = pj.y ;
     fPMT[j].z  = pj.z ;
  }

}

void PMTMap::~PMTMap(){
  delete [] fPMT ; 
  fPMT = NULL ;
}

void PMTMap::Dump(){
   
   PMT* pj = NULL ;
   for( Int_t j = 0 ; j < fEntries ; j++ ){
       pj = fPMT[j] ;
       cout << Form(" %d %f %f %f " , pj->id , pj->x , pj->y , pj->z ) << endl ;
   }
}


void PMTMap::Test( const char* path ){
  PMTMap* pm = new PMTMap();
  pm->Load( path );
  pm->Dump();
  
}
