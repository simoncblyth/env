
void tgeo_walk(){

    TGeoVolume* tv = gGeoManager->GetTopVolume();
    TGeoNode* tn = gGeoManager->GetTopNode();
    walk( tn , -1 );

}



void walk( TGeoNode* n , Int_t j ){

   TGeoVolume* v = n->GetVolume();
   cout << " j:" << j << " n:" << n->GetName() << " v:" << v->GetName() ;   
   TObjArray* a  = v->GetNodes() ;
   Int_t nn = a == NULL ? 0 : a->GetEntries() ;
   cout << " nn:" << nn << endl ;

   for(Int_t i=0 ; i < nn ; ++i ) walk( v->GetNode(i) , i ); 
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



