
#include <iostream>
#include "ZMQRoot.hh"
#include "TNamed.h"

int main() {

    TNamed* ptn = new TNamed("test", "test");
    ZMQRoot* pzmq = new ZMQRoot("ZMQ_TEST_SVR");
    TNamed& tn = *ptn;
    ZMQRoot& zmq = *pzmq;

    zmq.SendObject(&tn);

    TObject* tmpobj = zmq.ReceiveObject();
    std::cout << tmpobj << std::endl;
    tmpobj->Print();
}
