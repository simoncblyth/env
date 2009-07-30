readtxtfile(){
    
  //
  //   read a text file, containing numbers separated by commas and parse them into a usable form
  //   
  //  
    
  ifstream in("08A.ASC");
  char line[100];
  while( in.getline( line, 100  ) ){
	  
	 TString t = line ;
	 TObjArray* arr = t.Tokenize(",");
	 Int_t n = arr->GetEntries();
     for(Int_t i=0 ; i < n ; ++i ){
        TObjString* tob =(TObjString*)arr->At(i);
        TString tt = tob->GetString();
        Float_t ft = tt.Atof() ;
        cout << "[" << i << "][" << tt  << "][" << ft << "]"   ;
     }
     cout << endl ;
  }
}

