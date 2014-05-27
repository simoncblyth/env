
#include "ZMQRoot.hh"
#include "TNamed.h"

int main() {

    TNamed tn("test", "test");
    ZMQRoot zmq("ZMQ_TEST_SVR");

    zmq.SendObject(&tn);
}
