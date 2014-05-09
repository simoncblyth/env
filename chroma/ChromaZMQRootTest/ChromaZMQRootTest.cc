/*

::

    (chroma_env)delta:ChromaZMQRootTest blyth$ CHROMA_CLIENT_CONFIG=tcp://localhost:5555 ./ChromaZMQRootTest
    ZMQRoot::ZMQRoot envvar [CHROMA_CLIENT_CONFIG] config [tcp://localhost:5555] 
    ChromaPhotonList::Print  [1]
    ZMQRoot::SendObject sent bytes: 217 
    ZMQRoot::ReceiveObject received bytes: 217 
    ZMQRoot::ReceiveObject reading TObject from the TMessage 
    ZMQRoot::ReceiveObject returning TObject 
    ChromaZMQRootTest(32792,0x7fff7ab15310) malloc: *** error for object 0x7fa038d00398: pointer being freed was not allocated
    *** set a breakpoint in malloc_error_break to debug
    Abort trap: 6


*/


#include "ZMQRoot.hh"
#include "ChromaPhotonList.hh"
#include <iostream>

int main (int argc, char *argv[])
{

   ZMQRoot* q = new ZMQRoot("CHROMA_CLIENT_CONFIG") ;

   ChromaPhotonList* cpl, cpl2 ; 

   cpl = new ChromaPhotonList ;
   cpl->AddPhoton( 1.,1.,1.,  2.,2.,2.,  3.,3.,3.,  0., 100., 101 ); 
   cpl->Print();

   q->SendObject(cpl); 


   TObject* obj = q->ReceiveObject();
   std::cout << "obj received" << std::endl ;  

   obj->Print();

   //cpl2 = (ChromaPhotonList*)obj;
   //cpl2->Print();


   return 0 ; 
}

