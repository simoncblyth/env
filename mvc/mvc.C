/*

  Cocoa Design Patterns ...  a comparison of Qt signal/slots with ObjC message passing 
     http://safari.informit.com/9780321591210/ch17lev1sec4

   Implementing ObjC style messaging with S&S 
      http://lists.puremagic.com/pipermail/digitalmars-d/2006-September/008403.html


   Cocoa NSControl has target and action, the target object method specified by the action is 
   invoked when a control (eg NSButton) is triggered... the action method in the target 
   has a single argument .. the pointer to the sender object.   

   Target-Action Design pattern
      http://developer.apple.com/documentation/Cocoa/Conceptual/ObjectiveC/Articles/chapter_10_section_4.html#//apple_ref/doc/uid/TP30001163-CH23-TPXREF132



   root:TQObjSender   used to delegate TQObject functionality 
        used by the RQ_OBJECT.h macro, to allow classes not inheriting from TQObject 
        to take part in signals/slots 


   TQObject.cxx

void *gTQSender; // A pointer to the object that sent the last signal.
                 // Getting access to the sender might be practical
                 // when many signals are connected to a single slot.


#pragma link C++ global gTQSender;
#pragma link C++ global gTQSlotParams;




*/



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
      void NotifySender(Int_t); // *SIGNAL*  
};

void M::SetValue(Int_t val ){
    if( val != fValue ){
          fValue = val ;
          Emit("SetValue(Int_t)", val ); 
     }
}

void M::NotifySender(Int_t dummy){
     Emit("NotifySender(Int_t)", Int_t(this) ); 
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
      void NotifySender(Int_t); // *SIGNAL* 
};

void V::SetValue(Int_t val){
   cout << "V::SetValue was notified " << val  << endl ;
}

void V::NotifySender(Int_t addr){
   cout << "V::NotifySender was notified " << addr  << endl ;
   vm = (M*)addr ;
   vm->Print();
   
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
   m->Connect("NotifySender(Int_t)", "V", v , "NotifySender(Int_t)" );
  // Bool_t Connect(const char *signal, const char *receiver_class, void *receiver, const char *slot);


   // now the view gets notified of changes to the model, 
   // although class M knows nothing of class V  
   m->SetValue(1); m->NotifySender(0) ;
   m->SetValue(3); m->NotifySender(0) ;


}


