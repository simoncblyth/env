
//#include "TROOT.h"
//#include "TRint.h"
#include "TApplication.h"

#include "EvMQ.h"
#include <unistd.h>
#include <stdio.h>




int main(int argc, char **argv)
{
    printf("starting... %s\n", argv[0]);
    extern char **environ;
    int e = 0;
    while (environ[e] != NULL) {
    	printf("%s\r\n", environ[e]);
    	e++;
    }
    
    TApplication *theApp = new TApplication("ROOT example", &argc, argv );
    
    
    EvMQ* emq = new EvMQ();
 
    
    
    theApp->Run();
    return(0);
}





