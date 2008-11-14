/*
 
//
//  .L GeoMap.C                                     
//
// root [1]  GeoMap* gm = new GeoMap("Aberdeen_World.root");
// Info: TGeoManager::Import : Reading geometry from file: Aberdeen_World.root
// Info in <TGeoManager::CloseGeometry>: Geometry loaded from file...
// Info in <TGeoManager::SetTopVolume>: Top volume is World. Master volume is World
// Info in <TGeoManager::CloseGeometry>: Voxelization retrieved from file
// Info in <TGeoManager::BuildCache>: --- Maximum geometry depth is 100
// Info in <TGeoManager::CloseGeometry>: 286 nodes/ 20 volume UID's in VGM Root geometry
// Info in <TGeoManager::CloseGeometry>: ----------------modeler ready----------------
//
// root [2] gm->GetVol("World_1")->Draw("ogle")
// --- Drawing      143 nodes with 4 visible levels
// <TCanvas::MakeDefCanvas>: created default TCanvas with name c1
//
//   regular expression controlled colouring 
//
// root [3] gm->SetLineColor("^.*Gas.*$", kRed )
// root [4] gm->SetLineColor("^.*1\\.5.*$", kBlue )
// root [5] gm->SetLineColor("^.*Tube.*$", kGreen )
//
//   and visibility ... play hide and seek :
//
//  gm->SetVisibility("^.*$",kFALSE)        // hide all volumes
//
//  gm->SetVisibility("^.*1\\.5.*", kTRUE ) // just the 1.5m 
//  gm->SetVisibility("^.*Scin.*$",kTRUE)   // add Scintillators
//  gm->SelectKeys("^.*Scin.*$")->ls()      // list the keys
//
//  Make volumes with duplication issues red and visible
//
//  gm->SetLineColor("^.*x$",kRed)
//  gm->SetVisibility("^.*x$",kTRUE)
// 
//  Access individual volumes also :
// 
//  gm->GetVol("Door_0")->SetVisibility(kTRUE)
//  gm->GetVol("Door_0")->SetLineColor(kGreen)
//   


   PMT names should be renamed for simpler sorting ... PMT_00 etc..

root [2] gm->SelectKeys("PMT.*")->ls() 
OBJ: TList      TList   Doubly linked list : 0
 OBJ: TObjString        PMT_0   Collectable string class : 0 at: 0xa16d7f8
 OBJ: TObjString        PMT_1   Collectable string class : 0 at: 0xa1191b0
 OBJ: TObjString        PMT_10  Collectable string class : 0 at: 0xa119150
 OBJ: TObjString        PMT_11  Collectable string class : 0 at: 0xa16bae8
 OBJ: TObjString        PMT_12  Collectable string class : 0 at: 0xa16e810
 OBJ: TObjString        PMT_13  Collectable string class : 0 at: 0xa16d778

*/

class TGeoNode ;

class GeoMap {

   static TGeoNode* Import( const TString& filepath , const TString& vname ); 

public:
   GeoMap(TGeoNode* top=NULL);
   ~GeoMap();
   
   Bool_t HasKey(const TString& key);
   const char* UniqueKey(const TString& key);
   TMap* GetMap();

   // caution these are accessed by the key (usually the same as node name, but not when have dupes)
   TGeoNode* GetNod( const TString& key );
   TGeoVolume* GetVol( const TString& key );

   TList* SelectVol( const TString& patn );
   TList* SelectKeys( const TString& patn );
   TList* SelectKeys( const TString& patn, Bool_t viz );
   TList* SelectKeys( const TString& patn, enum EColor col );

   // the patn is a regular expression selection one or more (or no) volumes, see TRegexp for the syntax
   void SetLineColor(const TString& patn , enum EColor col );
   void SetVisibility(const TString& patn , Bool_t viz );
   void SetTransparency(const TString& patn , Char_t transparency );

   void Display(const TString& key);
   void Dump(TGeoNode* node , const TString& path="");
//   void ImportVolume(TString filepath, TString vname="World");

   void SetPMTHit(const TString& pmtnos, Double_t hitsize);
   void ResetBox(const TString& vkey, Double_t sx, Double_t sy, Double_t sz, EColor color);
   void ResetPMT(void);

   void Refresh(void);
   
private:
  // void Import(TString filepath);
   void Walk( TGeoNode* node , const TString& path="" );

   TMap* fMap ;
};



GeoMap* GeoMap::Import( const TString& filepath ){
    TGeoManager::Import( filepath);
    TGeoNode* tn = gGeoManager->GetTopNode();
    return new GeoMap( tn );
}


GeoMap::GeoMap(TGeoNode* top ) : fMap(NULL) {
    TGeoNode* node = top == NULL ?  gGeoManager->GetNode(0) : top ;
    fMap = new TMap ;
    Walk( node );
}

GeoMap::~GeoMap(){
  delete fMap ;
}

TMap* GeoMap::GetMap(){ 
  return fMap ; 
}

/*
void GeoMap::ImportVolume(TString filepath, TString vname ){
    TGeoVolume* top = TGeoVolume::Import( filepath , vname );
    gGeoManager->SetTopVolume( top );
    gGeoManager->CloseGeometry();
    CreateMap();
}
*/

Bool_t GeoMap::HasKey( const TString& key ){
   //cout << "HasKey[" << key << "]" << endl ;
   return fMap(key) != NULL ;
}

const char* GeoMap::UniqueKey( const TString& key ){ 
    // length check to stop recursive infinite loops 
     if ( key.Length() < 100 && !HasKey(key)){
        return key.Data() ;
     } else {
        TString nk = Form("%s%s" , key.Data(), "x" );
        return  UniqueKey( nk ); 
     }
}


TGeoNode* GeoMap::GetNod( const TString& key ){
   return (TGeoNode*)fMap(key);
}

TGeoVolume* GeoMap::GetVol( const TString& key ){
   TGeoNode* node = GetNod(key);
   return node==NULL ? NULL : node->GetVolume();
}


TList* GeoMap::SelectKeys( const TString& patn ){
   
   TRegexp re(patn);
   TIter next(fMap);
   TObjString* k = NULL ;
   Ssiz_t* len = new Ssiz_t ; 

   TList* sel = new TList ;

   while((  k = (TObjString*)next() )){
      TString key = k->GetString();
      if( re.Index(key,len) != kNPOS ){
           sel->Add( k ); 
      } 
   }

   sel->Sort();
   return sel ;
}

TList* GeoMap::SelectVol( const TString& patn ){

   TList* sel = SelectKeys(patn);
   TIter next(sel);
   TObjString* k = NULL ;

   TList* vs = new TList ;
   while((  k = (TObjString*)next() )){
       TGeoVolume* v = GetVol(k->GetString());
       vs->Add( v ); 
   }
   return vs ;
}


TList* GeoMap::SelectKeys( const TString& patn, Bool_t viz ){
    
   TList* sel = SelectKeys(patn);
   TIter next(sel);
   TObjString* k = NULL ;

   TList* nel = new TList ;
   while((  k = (TObjString*)next() )){
       TGeoVolume* v = GetVol(k->GetString());
       if( v->IsVisible() == viz ){
          nel->Add( k ); 
       } 
   }

   nel->Sort();
   return nel ;
}

TList* GeoMap::SelectKeys( const TString& patn , enum  EColor col ){
    
    
}

void GeoMap::SetVisibility(const TString& patn , Bool_t viz ){

   TList* sel = SelectKeys(patn);
   TIter next(sel);
   TObjString* key = NULL ;

   while(( key = (TObjString*)next() )){
       TGeoVolume* v = GetVol(key->GetString());
       v->SetVisibility( viz );
   }

}

void GeoMap::SetLineColor(const TString& patn , enum EColor col ){

   TList* sel = SelectKeys(patn);
   TIter next(sel);
   TObjString* key = NULL ;

   while(( key = (TObjString*)next() )){
       TGeoVolume* v = GetVol(key->GetString());
       v->SetLineColor( col );
   }
}

void GeoMap::SetTransparency(const TString& patn , Char_t tran ){

   TList* sel = SelectKeys(patn);
   TIter next(sel);
   TObjString* key = NULL ;

   while(( key = (TObjString*)next() )){
       TGeoVolume* v = GetVol(key->GetString());
       v->SetTransparency( tran );
   }

}


void GeoMap::Display( const TString& key ){

   TGeoNode* node = GetNod(key);
   Dump( node ); 
}

void GeoMap::Dump( TGeoNode* node , const TString& path ){

   TString name=node->GetName();
   cout << path << " [" << name << "]" << endl ;

   TString p = Form("%s/%s", path.Data(),name.Data());
   TGeoVolume* vol = node->GetVolume();
   TObjArray* a  = vol->GetNodes() ;
   Int_t nn = a == NULL ? 0 : a->GetEntries() ;
   for(Int_t i=0 ; i < nn ; ++i ) Dump( vol->GetNode(i) , p ); 

}

void GeoMap::Walk( TGeoNode* node, const TString& path ){

   // recursive tree walk, creating the map of nodes

   TString name=node == NULL ? "NULL" : node->GetName();
   //cout << path << " [" << name << "]" << endl ;
   TString key = UniqueKey( name );
   fMap->Add( new TObjString(key) , node );

   TString p = Form("%s/%s", path.Data(),name.Data());
   TGeoVolume* vol = node->GetVolume();
   TObjArray* a  = vol->GetNodes() ;
   Int_t nn = a == NULL ? 0 : a->GetEntries() ;
   for(Int_t i=0 ; i < nn ; ++i ) Walk( vol->GetNode(i) , p ); 
}

void GeoMap::ResetBox(const TString& vkey, Double_t sx, Double_t sy, Double_t sz, EColor color){

	// Replace PMT geometry using ReplaceNode() method



	TGeoVolume* v = GetVol(vkey);
	TGeoBBox* newbox = v->GetShape();
	newbox->SetBoxDimensions(sx,sy,sz);
	v->SetLineColor(color);

}

void GeoMap::SetPMTHit(Int_t pmtnos,Double_t hitsize ){

	// sets the size of box representing PMT response
	
	TString pmt="PMT_";
	TString pmtno = Form("%i",pmtnos);
	pmt += pmtno;

	// How to add hitsize? re-dim an small box( pixel) on PMT and accecss it the pixel
	// The larger the hitsize, the more pixels.
	//
	// sets the charge deposited in PMT
	//
	// represent the hit pattern in PMT with
	// The more charges, the larger the box
	//
	// Max. hitsize is 5. and min. is 0
	// In the future, we can use 5*(SPE charge)/(largest charge) to setting the max.
	// The charge can be decided by the mim likelihood function:see DocDB by Jun.

	Double_t sx = hitsize ;
	Double_t sy = 0.1 ;
	Double_t sz = hitsize ;
	//replace an square of the hited PMT 
	ResetBox(pmt,sx,sy,sz,kGreen);
	cout << "hit on " << pmt << "!!! The relative response to all PMTs is " << hitsize*20 << " %"<< endl;
}

void GeoMap::ResetPMT(void){

	
	// looping all PMTs and setting the dim of the PMT box to the default (5,0.1,5)
	// PMT_0 to PMT_15, so the Int_t i counter start from 0 and <16 below
	for(Int_t i=0;i<16;i++){
		TString pmt = "PMT_";
		TString pmtno = Form("%i",i);
		pmt += pmtno;
		//cout << "Resetting " << pmt << endl;
		ResetBox(pmt,5,0.1,5,kMagenta);
		//cout << " Reset "<< pmt << " already" << endl;
	}
	Refresh();
	cout << "   Reset PMT already " << endl;

}


void GeoMap::Refresh(void){

	//refresh the display
	
	//CreateMap();
	TGeoVolume* top = GetVol("World_1");
	top->Draw("ogle");
	
}
