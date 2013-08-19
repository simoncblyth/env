/*

http://www.mcs.anl.gov/~kazutomo/rdtsc.html


Compile and run::

    [blyth@belle7 assembly]$ cc rdtsc.c -o rdtsc && ./rdtsc
    1947319215609510
    1947319215609636
    126
    [blyth@belle7 assembly]$ cc rdtsc.c -o rdtsc && ./rdtsc
    1947342600261486
    1947342600261612
    126


Look at the assembly code::

    [blyth@belle7 assembly]$ cc -S rdtsc.c
    [blyth@belle7 assembly]$ cat rdtsc.s

    [blyth@belle7 assembly]$ cc -S -fverbose-asm rdtsc.c
    [blyth@belle7 assembly]$ cat rdtsc.s



*/

#include <stdio.h>
#include "rdtsc.h"

int main(int argc, char* argv[])
{
  unsigned long long a,b;

  a = rdtsc();
  b = rdtsc();

  printf("%llu\n", a);
  printf("%llu\n", b);
  printf("%llu\n", b-a);
  return 0;
}
