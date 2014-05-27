

#include "ZMQRoot.hh"
#include "TNamed.h"

int main() {

    TNamed* ptn = NULL;
    ZMQRoot* zmq = new ZMQRoot("ZMQ_TEST_SVR");

    TObject* ptmp = NULL;
    while(true) {
        ptmp = zmq->ReceiveObject();
        if (!ptmp) {
            continue;
        }
        ptmp->Print();
        zmq->SendObject(ptmp);
    }
}
