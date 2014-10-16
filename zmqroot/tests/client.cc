#include "ZMQRoot/ZMQRoot.hh"
#include "TNamed.h"

int main() {
    ZMQRoot* zmq = new ZMQRoot("ZMQROOT_TEST_FRONTEND",'Q');
    while(true){ 
        TNamed tn("test", "test");
        zmq->SendObject(&tn);
        TObject* obj = zmq->ReceiveObject();
        obj->Print();
    }
}
