//
/ 
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
//

class GeoMap {

public:
   GeoMap(const char* path);
   ~GeoMap();
   
   Bool_t HasKey(TString key);
   TString UniqueKey(TString key);
   TMap* GetMap();

   // caution these are accessed by the key (usually the same as node name, but not when have dupes)
   TGeoNode* GetNod( TString key );
   TGeoVolume* GetVol( TString key );
   TList* SelectVol( TString patn );

   // the patn is a regular expression selection one or more (or no) volumes, see TRegexp for the syntax
   void SetLineColor(TString patn , enum EColor col );
   void SetVisibility(TString patn , Bool_t viz );

   void Display(TString key);
   void Dump(TGeoNode* node , TString path="");
   void ImportVolume(TString filepath, TString vname="World");

   void SetPMTHit(TString pmtnos, Double_t hitsize);
   void HitPattern(Double_t hitsize);
   
private:
   void Import(TString filepath);
   void CreateMap();
   void Walk( TGeoNode* node , TString path );

   TMap* fMap ;
};


GeoMap::GeoMap(const char* path) : fMap(NULL) {
   if( path != NULL ) Import( path );
}

GeoMap::~GeoMap(){
  delete fMap ;
}

TMap* GeoMap::GetMap(){ 
    return fMap ; 
}

void GeoMap::Import(TString filepath){

    TGeoManager::Import( filepath);
    CreateMap();
}


void GeoMap::CreateMap(){
    fMap = new TMap ;
    TGeoNode* tn = gGeoManager->GetTopNode();
    Walk( tn , "" );
}


void GeoMap::ImportVolume(TString filepath, TString vname ){

    TGeoVolume* top = TGeoVolume::Import( filepath , vname );
    gGeoManager->SetTopVolume( top );
    gGeoManager->CloseGeometry();
    CreateMap();
}



Bool_t GeoMap::HasKey( TString key ){
   //cout << "HasKey[" << key << "]" << endl ;
   return fMap(key) != NULL ;
}

TString GeoMap::UniqueKey( TString key ){ 
    // length check to stop recursive infinite loops 
     return key.Length() < 100 && !HasKey(key) ? key : UniqueKey( Form("%s%s", key.Data(), "x")); 
}


TGeoNode* GeoMap::GetNod( TString key ){
   return (TGeoNode*)fMap(key);
}

TGeoVolume* GeoMap::GetVol( TString key ){
   TGeoNode* node = GetNod(key);
   return node==NULL ? NULL : node->GetVolume();
}


TList* GeoMap::SelectKeys( TString patn ){
   
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

   return sel ;
}


void GeoMap::SetVisibility(TString patn , Bool_t viz ){

   TList* sel = SelectKeys(patn);
   TIter next(sel);
   TObjString* key = NULL ;

   while(( key = (TObjString*)next() )){
       TGeoVolume* v = GetVol(key->GetString());
       v->SetVisibility( viz );
   }

}

void GeoMap::SetLineColor(TString patn , enum EColor col ){

   TList* sel = SelectKeys(patn);
   TIter next(sel);
   TObjString* key = NULL ;

   while(( key = (TObjString*)next() )){
       TGeoVolume* v = GetVol(key->GetString());
       v->SetLineColor( col );
   }

}

void GeoMap::Display( TString key ){

   TGeoNode* node = GetNod(key);
   Dump( node ); 
}

void GeoMap::Dump( TGeoNode* node , TString path ){

   TString name=node->GetName();
   cout << path << " [" << name << "]" << endl ;

   TString p = Form("%s/%s", path.Data(),name.Data());
   TGeoVolume* vol = node->GetVolume();
   TObjArray* a  = vol->GetNodes() ;
   Int_t nn = a == NULL ? 0 : a->GetEntries() ;
   for(Int_t i=0 ; i < nn ; ++i ) Dump( vol->GetNode(i) , p ); 

}

void GeoMap::Walk( TGeoNode* node, TString path ){

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

void GeoMap::ReplacePMT(TString pmt, Double_t sx, Double_t sy, Double_t sz, EColor color){

	// Replace PMT geometry using ReplaceNode() method



	TGeoVolume* world = GetVol("World_1");
	TGeoNode* oldpmtnode = GetNod(pmt);
	TGeoNode* newpmtnode = world->ReplaceNode(oldpmtnode, new TGeoBBox(sx,sy,sz));
	TGeoVolume* newpmtvol = newpmtnode->GetVolume();
	newpmtvol->SetLineColor(color);

}

void GeoMap::SetPMTHit(Int_t pmtnos,Double_t hitsize ){

	// sets the size of box representing PMT response
	
	TString pmt="PMT_";
	TString pmtno = Form("%i",pmtnos);
	pmt += pmtno;

	// How to add hitsize? create an small box( pixel) on PMT and accecss it the pixel
	// The larger the hitsize, the more pixels.
	HitPattern(pmt,hitsize);

	
}


void GeoMap::HitPattern(TString pmt, Float_t hitsize){

	// sets the charge deposited in PMT

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
	ReplacePMT(pmt,sx,sy,sz,kGreen);
	//ReplacePMT(pmt,sx,sy,sz,kRed);
	cout << "hit on " << pmt << "!!! The relative response to all PMTs is " << hitsize*20 << " %"<< endl;
}

void GeoMap::ResetPMT(void){

	delete fMap;		// release the dynamic memory
	CreateMap();		// make sure use the latest mapping. later
	
	// looping all PMTs and kill the volume inside them
	// PMT_0 to PMT_15, so the Int_t i counter start from 0 and <16 below
	for(Int_t i=0;i<16;i++){
		TString pmt = "PMT_";
		TString pmtno = Form("%i",i);
		pmt += pmtno;
		//cout << "Resetting " << pmt << endl;
		ReplacePMT(pmt,5,0.1,5,kMagenta);
		//cout << " Reset "<< pmt << " already" << endl;
	}

	cout << "   Reset PMT already " << endl;
	delete fMap;
	CreateMap();		// make sure use the latest mapping in the future
}
/*
void GeoMap::ResetSubVolume(TGeoNode* node){

	//kill all the sub volumes
	TGeoVolume* vol = node->GetVolume();
	TObjArray* a = vol->GetNodes();
	Int_t nn = a == NULL ? 0 : a->GetEntries();
	for(Int_t i=0; i< nn ; i++) {
		TGeoVolume* world = GetVol("World_1");
		ResetSubVolume(vol->GetNode(i));
		cout <<"!!!!!!!!!!!D"<< endl;
	}
	cout << "!!!!!!!!!!!!E"<< endl;

}
*/


// useless in Display.C after using ReplaceNode() method
void GeoMap::Refresh(void){

	//refresh the display
	
	//CreateMap();
	TGeoVolume* top = GetVol("World_1");
	top->Draw("ogle");
	
}
