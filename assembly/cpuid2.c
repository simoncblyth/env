/*

* http://newbiz.github.io/cpp/2010/12/20/Playing-with-cpuid.html

::

    [blyth@belle7 assembly]$ g++ cpuid2.c -o cpuid2
    [blyth@belle7 assembly]$ ./cpuid2
    Processor name: GenuineIntel

    Max cpuid call: 10

    Processor features:
      FPU   = true
      VME   = true
      DE    = true
      PSE   = true
      TSC   = true
      MSR   = true
      PAE   = true
      MCE   = true
   ...



*/



#if defined(__GNUC__)
    #include <stdint.h>
#elif defined(_WIN32)
    #include <intrin.h>
    typedef unsigned __int32 uint32_t;
#endif



static void call_cpuid()
{
  // http://newbiz.github.io/cpp/2010/12/20/Playing-with-cpuid.html
  // GCC won't allow us to clobber EBX since its used to store the GOT. So we need to
  // lie to GCC and backup/restore EBX without declaring it as clobbered.

  __asm__ volatile( "pushl %ebx   \n\t"
                    ".byte 0x0f, 0xa2 \n\t"
                    "movl %ebx, %1\n\t"
                    "popl %ebx    \n\t"
                    :::"%eax", "%ecx", "%edx");    // ebx not declared clobbered
}



/**
 * Calls cpuid with op and store results of eax,ebx,ecx,edx
 * \param op cpuid function (eax input)
 * \param eax content of eax after the call to cpuid
 * \param ebx content of ebx after the call to cpuid
 * \param ecx content of ecx after the call to cpuid
 * \param edx content of edx after the call to cpuid
 */
void cpuid( uint32_t op, uint32_t& eax, uint32_t& ebx, uint32_t& ecx, uint32_t& edx )
{
#if defined(__GNUC__)
  // GCC won't allow us to clobber EBX since its used to store the GOT. So we need to
  // lie to GCC and backup/restore EBX without declaring it as clobbered.
  asm volatile( "pushl %%ebx   \n\t"
                "cpuid         \n\t"
                "movl %%ebx, %1\n\t"
                "popl %%ebx    \n\t"
                : "=a"(eax), "=r"(ebx), "=c"(ecx), "=d"(edx)
                : "a"(op)
                : "cc" );
#elif defined(_WIN32)
  // MSVC provides a __cpuid function
  int regs[4];
  __cpuid( regs, op );
  eax = regs[0];
  ebx = regs[1];
  ecx = regs[2];
  edx = regs[3];
#endif
}

/**
 * Retrieve the maximum function callable using cpuid
 */
uint32_t cpuid_maxcall()
{
  uint32_t eax, ebx, ecx, edx;
  cpuid( 0, eax, ebx, ecx, edx );
  return eax;
}

/**
 * Reference:
 * http://datasheets.chipdb.org/Intel/x86/CPUID/24161821.pdf
 * http://www.flounder.com/cpuid_explorer2.htm
 */
enum CpuidFeatures
{
  FPU   = 1<< 0, // Floating-Point Unit on-chip
  VME   = 1<< 1, // Virtual Mode Extension
  DE    = 1<< 2, // Debugging Extension
  PSE   = 1<< 3, // Page Size Extension
  TSC   = 1<< 4, // Time Stamp Counter
  MSR   = 1<< 5, // Model Specific Registers
  PAE   = 1<< 6, // Physical Address Extension
  MCE   = 1<< 7, // Machine Check Exception
  CX8   = 1<< 8, // CMPXCHG8 Instruction
  APIC  = 1<< 9, // On-chip APIC hardware
  SEP   = 1<<11, // Fast System Call
  MTRR  = 1<<12, // Memory type Range Registers
  PGE   = 1<<13, // Page Global Enable
  MCA   = 1<<14, // Machine Check Architecture
  CMOV  = 1<<15, // Conditional MOVe Instruction
  PAT   = 1<<16, // Page Attribute Table
  PSE36 = 1<<17, // 36bit Page Size Extension
  PSN   = 1<<18, // Processor Serial Number
  CLFSH = 1<<19, // CFLUSH Instruction
  DS    = 1<<21, // Debug Store
  ACPI  = 1<<22, // Thermal Monitor & Software Controlled Clock
  MMX   = 1<<23, // MultiMedia eXtension
  FXSR  = 1<<24, // Fast Floating Point Save & Restore
  SSE   = 1<<25, // Streaming SIMD Extension 1
  SSE2  = 1<<26, // Streaming SIMD Extension 2
  SS    = 1<<27, // Self Snoop
  HTT   = 1<<28, // Hyper Threading Technology
  TM    = 1<<29, // Thermal Monitor
  PBE   = 1<<31, // Pend Break Enabled
};
/**
 * This will retrieve the CPU features available
 * \return The content of the edx register containing available features
 */
uint32_t cpuid_features()
{
  uint32_t eax, ebx, ecx, edx;
  cpuid( 1, eax, ebx, ecx, edx );
  return edx;
}

/**
 * Reference:
 * http://datasheets.chipdb.org/Intel/x86/CPUID/24161821.pdf
 * http://www.flounder.com/cpuid_explorer2.htm
 */
enum CpuidExtendedFeatures
{
  SSE3  = 1<< 0, // Streaming SIMD Extension 3
  MW    = 1<< 3, // Mwait instruction
  CPL   = 1<< 4, // CPL-qualified Debug Store
  VMX   = 1<< 5, // VMX
  EST   = 1<< 7, // Enhanced Speed Test
  TM2   = 1<< 8, // Thermal Monitor 2
  L1    = 1<<10, // L1 Context ID
  CAE   = 1<<13, // CompareAndExchange 16B
};
/**
 * This will retrieve the extended CPU features available
 * \return The content of the ecx register containing available extended features
 */
uint32_t cpuid_extended_features()
{
  uint32_t eax, ebx, ecx, edx;
  cpuid( 1, eax, ebx, ecx, edx );
  return ecx;
}

/**
 * Retrieve the processor name.
 * \param name Preallocated string containing at least room for 13 characters. Will
 *             contain the name of the processor.
 */
void cpuid_procname( char* name )
{
  name[12] = 0;
  uint32_t max_op;
  cpuid( 0, max_op, (uint32_t&)name[0], (uint32_t&)name[8], (uint32_t&)name[4] );
}



#include <cstdlib>
#include <iostream>

int main( int, char** )
{
  char procname[13];
  cpuid_procname(procname);
  std::cout << "Processor name: " << procname << std::endl;
  std::cout << std::endl;
  std::cout << "Max cpuid call: " << cpuid_maxcall() << std::endl;
  std::cout << std::endl;
  std::cout << "Processor features:" << std::endl;
  std::cout << "  FPU   = " << std::boolalpha << (bool)(cpuid_features() & FPU  ) << std::endl;
  std::cout << "  VME   = " << std::boolalpha << (bool)(cpuid_features() & VME  ) << std::endl;
  std::cout << "  DE    = " << std::boolalpha << (bool)(cpuid_features() & DE   ) << std::endl;
  std::cout << "  PSE   = " << std::boolalpha << (bool)(cpuid_features() & PSE  ) << std::endl;
  std::cout << "  TSC   = " << std::boolalpha << (bool)(cpuid_features() & TSC  ) << std::endl;
  std::cout << "  MSR   = " << std::boolalpha << (bool)(cpuid_features() & MSR  ) << std::endl;
  std::cout << "  PAE   = " << std::boolalpha << (bool)(cpuid_features() & PAE  ) << std::endl;
  std::cout << "  MCE   = " << std::boolalpha << (bool)(cpuid_features() & MCE  ) << std::endl;
  std::cout << "  CX8   = " << std::boolalpha << (bool)(cpuid_features() & CX8  ) << std::endl;
  std::cout << "  APIC  = " << std::boolalpha << (bool)(cpuid_features() & APIC ) << std::endl;
  std::cout << "  SEP   = " << std::boolalpha << (bool)(cpuid_features() & SEP  ) << std::endl;
  std::cout << "  MTRR  = " << std::boolalpha << (bool)(cpuid_features() & MTRR ) << std::endl;
  std::cout << "  PGE   = " << std::boolalpha << (bool)(cpuid_features() & PGE  ) << std::endl;
  std::cout << "  MCA   = " << std::boolalpha << (bool)(cpuid_features() & MCA  ) << std::endl;
  std::cout << "  CMOV  = " << std::boolalpha << (bool)(cpuid_features() & CMOV ) << std::endl;
  std::cout << "  PAT   = " << std::boolalpha << (bool)(cpuid_features() & PAT  ) << std::endl;
  std::cout << "  PSE36 = " << std::boolalpha << (bool)(cpuid_features() & PSE36) << std::endl;
  std::cout << "  PSN   = " << std::boolalpha << (bool)(cpuid_features() & PSN  ) << std::endl;
  std::cout << "  CLFSH = " << std::boolalpha << (bool)(cpuid_features() & CLFSH) << std::endl;
  std::cout << "  DS    = " << std::boolalpha << (bool)(cpuid_features() & DS   ) << std::endl;
  std::cout << "  ACPI  = " << std::boolalpha << (bool)(cpuid_features() & ACPI ) << std::endl;
  std::cout << "  MMX   = " << std::boolalpha << (bool)(cpuid_features() & MMX  ) << std::endl;
  std::cout << "  FXSR  = " << std::boolalpha << (bool)(cpuid_features() & FXSR ) << std::endl;
  std::cout << "  SSE   = " << std::boolalpha << (bool)(cpuid_features() & SSE  ) << std::endl;
  std::cout << "  SSE2  = " << std::boolalpha << (bool)(cpuid_features() & SSE2 ) << std::endl;
  std::cout << "  SS    = " << std::boolalpha << (bool)(cpuid_features() & SS   ) << std::endl;
  std::cout << "  HTT   = " << std::boolalpha << (bool)(cpuid_features() & HTT  ) << std::endl;
  std::cout << "  TM    = " << std::boolalpha << (bool)(cpuid_features() & TM   ) << std::endl;
  std::cout << std::endl;
  std::cout << "Processor extended features:" << cpuid_extended_features() << std::endl;
  std::cout << "  SSE3 = " << std::boolalpha << (bool)(cpuid_extended_features() & SSE3) << std::endl;
  std::cout << "  MW   = " << std::boolalpha << (bool)(cpuid_extended_features() & MW  ) << std::endl;
  std::cout << "  CPL  = " << std::boolalpha << (bool)(cpuid_extended_features() & CPL ) << std::endl;
  std::cout << "  VMX  = " << std::boolalpha << (bool)(cpuid_extended_features() & VMX ) << std::endl;
  std::cout << "  EST  = " << std::boolalpha << (bool)(cpuid_extended_features() & EST ) << std::endl;
  std::cout << "  TM2  = " << std::boolalpha << (bool)(cpuid_extended_features() & TM2 ) << std::endl;
  std::cout << "  L1   = " << std::boolalpha << (bool)(cpuid_extended_features() & L1  ) << std::endl;
  std::cout << "  CAE  = " << std::boolalpha << (bool)(cpuid_extended_features() & CAE ) << std::endl;

  call_cpuid(); 
  
  return EXIT_SUCCESS;
}


