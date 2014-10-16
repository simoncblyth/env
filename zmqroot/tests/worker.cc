#include "ZMQRoot/ZMQRoot.hh"
#include "TNamed.h"

int main() {
    ZMQRoot* zmq = new ZMQRoot("ZMQROOT_TEST_BACKEND",'P');
    TObject* obj = NULL;
    while(true) {
        obj = zmq->ReceiveObject();
        if (!obj) continue;
        obj->Print();
        zmq->SendObject(obj);
    }
}
