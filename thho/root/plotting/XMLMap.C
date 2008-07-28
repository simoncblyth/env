//
//   Exercise roots XML parsing capabilities to read a simple XML file (such
//   as MaterialProperties.xml)  and create a corresponding TMap 
//   keyed by the "element path" (assumed unique keys) with
//   objects comprised of TObjArrays of TObjStrings  
//
//   Interactive test :
//     .L XMLMap.C
//     TString path =  Form("%s/G4dyb/data/xml/MaterialProperties.xml",gSystem->Getenv("DYW")) ; XMLMap* xm = new XMLMap ; xm->Parse( path ); m = xm->fMap ;
//
//   Use the created map :   
//     ((TObjArray*)m("/SimplePMT/SimplePMT_MomentumBins"))->Print()
//     ((TObjArray*)m("/LS/LS_Reemission"))->Print() 
//
//   TODO :   
//      find out the desired form for making plots and create convenience
//      converters to go from the TObjArray of TObjStrings to the required
//      form
//
//
// g = m->MakeGraph(m,"/PMT_Property/QE_Momentum","/PMT_Property/QE",1.0, 1.0);
// 

class XMLMap {

   public:
     TMap* fMap ;
     
     XMLMap();
     ~XMLMap();
     void Parse(const char* filepath);


     TObjArray* GetArray(TString xmlpath);
     TGraphErrors* MakeGraph(TString xpath, TString ypath, Double_t wx , Double_t wy );
     
     void Walk( TXMLNode* base , TString path  );
     static void MPT();   

};


TObjArray* XMLMap::GetArray(TString xmlpath){
   TObjArray* a = ((TObjArray*)fMap(xmlpath)) ;
   return a ;
}



TGraphErrors* XMLMap::MakeGraph(TString xpath, TString ypath, Double_t wx, Double_t wy){

   TObjArray* xa = GetArray(xpath);
   TObjArray* ya = GetArray(ypath);
   
   Int_t nx = xa->GetEntries();
   Int_t ny = ya->GetEntries();
   
   if( nx != ny){
	   cout << "imcompatible with x and y entries!!!" << endl;
	   return NULL;
   }
   else{
	   cout << "The entries is " << nx << endl;
   }
	   
   TGraphErrors* g = new TGraphErrors(nx);
   for( Int_t ii = 0 ; ii < nx  ; ++ii ){ 
	   TString sx = ((TObjString*)xa->At(ii))->GetString();
	   TString sy = ((TObjString*)ya->At(ii))->GetString();

	   Double_t xx =atof(sx.Data())*wx;
	   Double_t yy =atof(sy.Data())*wy;
	   
		   
	   g->SetPoint( ii , xx , yy );
	   cout << " ii/sx/sy/xx/yy " << ii << "," << sx << "," << sy << "," << xx << "," << yy << ","  << endl ;
   }

   return g;
}	

void XMLMap::XMLMap(){
  fMap = NULL ;
}

void XMLMap::~XMLMap(){
}


void XMLMap::Walk( TXMLNode* base , TString path  ){
    
   for( TXMLNode* n = base->GetChildren() ; n != NULL ; n = n->GetNextNode()){
      TString nn = n->GetNodeName();
      TXMLNode::EXMLElementType t = n->GetNodeType();
      
      if( t == TXMLNode::kXMLElementNode ){
         TString p = Form( "%s/%s" , path.Data() , nn.Data() );
         cout << p  << endl ;
         Walk( n , p ) ;
         
      } else if ( t == TXMLNode::kXMLTextNode ){
         if(fMap == NULL ) fMap = new TMap ;

         TString s = base->GetText();
	 s.ReplaceAll( "\n" , " " );
         fMap->Add( new TObjString(path.Data()) , s.Tokenize(" ") );
      }
   }
}


void XMLMap::Parse( const char* filepath ){
   TDOMParser tdp ;
   tdp.ParseFile( filepath );
   TXMLDocument* doc = tdp.GetXMLDocument() ;
   TXMLNode* r = doc->GetRootNode() ;
   Walk( r , "" ); 
}


void XMLMap::MPT(){
   TString path = Form("%s/G4dyb/data/xml/MaterialProperties.xml",gSystem->Getenv("DYW")) ;
   XMLMap* xm = new XMLMap ;
   xm->Parse( path );
}




