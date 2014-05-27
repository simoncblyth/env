

#include "ZMQRoot.hh"
#include "TNamed.h"

int main() {

    TNamed* ptn;
    ZMQRoot zmq("ZMQ_TEST_SVR");

    TObject* ptmp = zmq.ReceiveObject();
    ptmp->Print();
}
