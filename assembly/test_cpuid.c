/*

   cc test_cpuid.c -o test_cpuid && ./test_cpuid


*/

#include <stdint.h>

static __inline__ void cpuid()
{

  // http://newbiz.github.io/cpp/2010/12/20/Playing-with-cpuid.html
  // GCC won't allow us to clobber EBX since its used to store the GOT. So we need to
  // lie to GCC and backup/restore EBX without declaring it as clobbered.

  uint32_t op;  // input:  eax
  uint32_t eax; // output: eax
  uint32_t ebx; // output: ebx
  uint32_t ecx; // output: ecx
  uint32_t edx; // output: edx

  __asm__ volatile( "pushl %%ebx   \n\t"
                    ".byte 0x0f, 0xa2 \n\t"
                    "movl %%ebx, %0\n\t"
                    "popl %%ebx    \n\t"
                    : "=a"(eax), "=r"(ebx), "=c"(ecx), "=d"(edx)
                    : "a"(op)
                    : "cc" );

}



int main(void)
{
   cpuid();
   return 0;
}
 

