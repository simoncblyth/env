
#include <RQ_OBJECT.h>

// Model 
class M {
   RQ_OBJECT("M")
   private:
      Int_t fValue ;
   public:
      M(){  fValue = 0 ; }
      Int_t GetValue(){ return fValue ; }
      void SetValue(Int_t ); // *SIGNAL*  
};

void M::SetValue(Int_t val ){
    if( val != fValue ){
          fValue = val ;
          Emit("SetValue(Int_t)", val ); 
     }
}

void M::Print(Option_t* option = "") const {
    cout << "M::  " << fValue << endl ;
}




// View 
class V {
   RQ_OBJECT("V")
   public:
      V(){}
      void SetValue(Int_t); // *SIGNAL* 
      void ValueDidChange(); // *SIGNAL* 
};

void V::SetValue(Int_t val){
   cout << "V::SetValue was notified " << val  << endl ;
}

void V::ValueDidChange(){
   cout << "V::ValueDidChange was notified " << endl ;
}


// Controller 
//class C {
//  public:
//    C(){}
//    void Hookup(M* m, V* v){
//    }
//};





M* m = NULL ;
V* v = NULL ;
//C* c = NULL ;


void mvc(){

   m = new M();
   v = new V();

   // connect the model to the view 
   m->Connect("SetValue(Int_t)",     "V", v , "SetValue(Int_t)" );
   m->Connect("SetValue(Int_t)",     "V", v , "ValueDidChange()" );
  // Bool_t Connect(const char *signal, const char *receiver_class, void *receiver, const char *slot);


   // now the view gets notified of changes to the model, 
   // although class M knows nothing of class V  
   m->SetValue(1);
   m->SetValue(3);


}


