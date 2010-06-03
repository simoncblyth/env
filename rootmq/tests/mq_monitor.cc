
//#include "TROOT.h"
#include "TRint.h"

#include "EvMQ.h"

int main(int argc, char **argv)
{
    TRint *theApp = new TRint("ROOT example", &argc, argv, NULL, 0);
    EvMQ* emq = new EvMQ();
    theApp->Run();
    return(0);
}





