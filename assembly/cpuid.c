/*

http://stackoverflow.com/questions/3113728/error-in-my-first-assembly-program-gcc-inline-assembly

Note that you need to compile with -fno-pic as EBX is reserved when PIC is enabled. (Either that or you need to take steps to save and restore EBX).

::

    [blyth@belle7 assembly]$ gcc -Wall -fno-pic cpuid.c -o cpuid
    [blyth@belle7 assembly]$ ./cpuid
    CPUID_getL1CacheSize = 64

*/
#include <stdio.h>

int CPUID_getL1CacheSize()
{
    int l1CacheSize = -1;

    asm ( "mov $5, %%eax\n\t"   // EAX=80000005h: L1 Cache and TLB Identifiers
          "cpuid\n\t"
          "mov %%eax, %0"       // eax into l1CacheSize 
          : "=r"(l1CacheSize)   // output 
          :                     // no input
          : "%eax", "%ebx", "%ecx", "%edx"  // clobbered registers
         );

    return l1CacheSize;
}

int main(void)
{
    printf("CPUID_getL1CacheSize = %d\n", CPUID_getL1CacheSize());
    return 0;
}

