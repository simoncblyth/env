// cc -g -o /tmp/gmalloctest gmalloctest.c


#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv) 
{
    unsigned *buffer = (unsigned *)malloc(sizeof(unsigned) * 100);
    unsigned i;

    for (i = 0; i < 200; i++) {
       buffer[i] = i;
    }

    for (i = 0; i < 200; i++) {
       printf ("%d  ", buffer[i]);
    }
}


/*

simon:gmalloctest blyth$ lldb /tmp/gmalloctest
(lldb) target create "/tmp/gmalloctest"
Current executable set to '/tmp/gmalloctest' (x86_64).
(lldb) env DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib
(lldb) r
GuardMalloc[sh-30107]: Allocations will be placed on 16 byte boundaries.
GuardMalloc[sh-30107]:  - Some buffer overruns may not be noticed.
GuardMalloc[sh-30107]:  - Applications using vector instructions (e.g., SSE) should work.
GuardMalloc[sh-30107]: version 27
GuardMalloc[arch-30107]: Allocations will be placed on 16 byte boundaries.
GuardMalloc[arch-30107]:  - Some buffer overruns may not be noticed.
GuardMalloc[arch-30107]:  - Applications using vector instructions (e.g., SSE) should work.
GuardMalloc[arch-30107]: version 27
Process 30107 launched: '/tmp/gmalloctest' (x86_64)
GuardMalloc[gmalloctest-30107]: Allocations will be placed on 16 byte boundaries.
GuardMalloc[gmalloctest-30107]:  - Some buffer overruns may not be noticed.
GuardMalloc[gmalloctest-30107]:  - Applications using vector instructions (e.g., SSE) should work.
GuardMalloc[gmalloctest-30107]: version 27
Process 30107 stopped
* thread #1: tid = 0x744e23, 0x0000000100000efc gmalloctest`main(argc=1, argv=0x00007fff5fbfee48) + 76 at gmalloctest.c:13, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x10034a000)
    frame #0: 0x0000000100000efc gmalloctest`main(argc=1, argv=0x00007fff5fbfee48) + 76 at gmalloctest.c:13
   10       unsigned i;
   11   
   12       for (i = 0; i < 200; i++) {
-> 13          buffer[i] = i;
   14       }
   15   
   16       for (i = 0; i < 200; i++) {
(lldb) p i
(unsigned int) $0 = 100
(lldb) 



*/
