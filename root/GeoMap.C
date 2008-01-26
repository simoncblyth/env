



// .L GeomMap.C
// GeoMap* gm = new GeoMap ;
// gm->Import("Aberdeen_World.root");
//

class GeoMap {


   
   void Import(TString filepath);
   Bool_t HasKey(TString key);
   TString UniqueKey(TString key);
   void Walk( TGeoNode* node , TString path );

   TMap* fMap ;
};




void GeoMap::Import(TString filepath){

     TGeoManager::Import(   filepath);
     
    TGeoVolume* tv = gGeoManager->GetTopVolume();
    TGeoNode* tn = gGeoManager->GetTopNode();
    Walk( tn , "" );

}

Bool_t GeoMap::HasKey( TString key ){
   return fMap(key) != NULL ;
}

TString GeoMap::UniqueKey( TString key ){ 
     return key.Length() < 100 && !HasKey(key) ? key : UniqueKey( Form("%s%s", key.Data(), "x")); 
}



void GeoMap::Walk( TGeoNode* node, TString path ){

   TString p = Form("%s/%s", path.Data(),node->GetName());
   TGeoVolume* vol = node->GetVolume();
   TObjArray* a  = vol->GetNodes() ;
   Int_t nn = a == NULL ? 0 : a->GetEntries() ;
   for(Int_t i=0 ; i < nn ; ++i ) Walk( vol->GetNode(i) , p ); 
}


void careful_walk( TGeoNode* n ){

  cout << " n:" ;
  if( n == NULL ){
     cout << "NULL" ;
  } else { 

     cout << n->GetName() ; 

     cout << " v:" ;

     TGeoVolume* v = n->GetVolume();
     if( v == NULL ){
        cout << "NULL" ;
     } else {
        cout << v->GetName() ;
     }

     TObjArray* a  = v->GetNodes() ;
     Int_t nn = a == NULL ? 0 : a->GetEntries() ;
  
     cout << " nn:" << nn << endl ;

     for(Int_t i=0 ; i < nn ; ++i ){

       cout << " i:" << i ;
       TGeoNode* ni = v->GetNode(i) ;
       if ( ni == NULL ){
          cout << " null " ;
       } else { 
          careful_walk( ni );
       }
     }
 }

}



