/*
*/

#include "ZMQRoot.hh"
#include "ChromaPhotonList.hh"
#include <iostream>

int main (int argc, char *argv[])
{

   ZMQRoot* q = new ZMQRoot("CHROMA_CLIENT_CONFIG") ;

   ChromaPhotonList* cpl = new ChromaPhotonList ;
   cpl->AddPhoton( 1.,1.,1.,  2.,2.,2.,  3.,3.,3.,  0., 100., 101 ); 
   cpl->AddPhoton( 1.,1.,1.,  2.,2.,2.,  3.,3.,3.,  0., 100., 101 ); 
   cpl->AddPhoton( 1.,1.,1.,  2.,2.,2.,  3.,3.,3.,  0., 100., 101 ); 
   cpl->AddPhoton( 1.,1.,1.,  2.,2.,2.,  3.,3.,3.,  0., 100., 101 ); 
   cpl->AddPhoton( 1.,1.,1.,  2.,2.,2.,  3.,3.,3.,  0., 100., 101 ); 
   cpl->AddPhoton( 1.,1.,1.,  2.,2.,2.,  3.,3.,3.,  0., 100., 101 ); 
   cpl->Print();

   q->SendObject(cpl); 


   TObject* obj = q->ReceiveObject();
   std::cout << "ReceiveObject" << std::endl ;  
   obj->Print();

   ChromaPhotonList* cpl2 = (ChromaPhotonList*)obj;
   cpl2->Print();
   cpl2->Details();


   return 0 ; 
}

