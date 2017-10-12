# === func-gen- : tools/nasm fgp tools/nasm.bash fgn nasm fgh tools
nasm-src(){      echo tools/nasm.bash ; }
nasm-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nasm-src)} ; }
nasm-vi(){       vi $(nasm-source) ; }
nasm-usage(){ cat << EOU


NASM - The Netwide Assembler
===============================

* http://www.nasm.us
* http://cs.lmu.edu/~ray/notes/nasmtutorial/


Observations

* OSX System Nasm too old to support macho64


Registers
------------

* http://cons.mit.edu/sp17/x86-64-architecture-guide.html

=========  ========================================  ======================
Register    Purpose                                   Saved across calls
=========  ========================================  ======================
%rax        temp register; return value               No
%rbx        callee-saved                              Yes
%rsp        stack pointer                             Yes
%rbp        callee-saved; base pointer                Yes
%rdi        used to pass 1st argument to functions    No
%rsi        used to pass 2nd argument to functions    No
%rdx        used to pass 3rd argument to functions    No
%rcx        used to pass 4th argument to functions    No
%r8         used to pass 5th argument to functions    No
%r9         used to pass 6th argument to functions    No
%r10-r11    temporary                                 No
%r12-r15    callee-saved registers                    Yes
=========  ========================================  ======================






::

    simon:nasm blyth$ nasm -hf
    ...
        -o outfile  write output to an outfile

        -f format   select an output format

        -l listfile write listing to a listfile

    ...

    valid output formats for -f are (`*' denotes default):
      * bin       flat-form binary files (e.g. DOS .COM, .SYS)
        ith       Intel hex
        srec      Motorola S-records
        aout      Linux a.out object files
        aoutb     NetBSD/FreeBSD a.out object files
        coff      COFF (i386) object files (e.g. DJGPP for DOS)
        elf32     ELF32 (i386) object files (e.g. Linux)
        elf64     ELF64 (x86_64) object files (e.g. Linux)
        elfx32    ELFX32 (x86_64) object files (e.g. Linux)
        as86      Linux as86 (bin86 version 0.3) object files
        obj       MS-DOS 16-bit/32-bit OMF object files
        win32     Microsoft Win32 (i386) object files
        win64     Microsoft Win64 (x86-64) object files
        rdf       Relocatable Dynamic Object File Format v2.0
        ieee      IEEE-695 (LADsoft variant) object file format
        macho32   NeXTstep/OpenStep/Rhapsody/Darwin/MacOS X (i386) object files
        macho64   NeXTstep/OpenStep/Rhapsody/Darwin/MacOS X (x86_64) object files
        dbg       Trace of all info passed to output stage
        elf       ELF (short name for ELF32)
        macho     MACHO (short name for MACHO32)
        win       WIN (short name for WIN32)



EOU
}

nasm-ver(){ echo 2.13.01 ; }
nasm-nam(){ echo nasm-$(nasm-ver) ; }
nasm-url(){ echo http://www.nasm.us/pub/nasm/releasebuilds/$(nasm-ver)/nasm-$(nasm-ver).tar.gz ; }    
nasm-dir(){ echo $(local-base)/env/tools/nasm/$(nasm-nam) ; }
nasm-cd(){  cd $(nasm-dir); }

nasm-tcd(){ cd $(env-home)/tools/nasm ; }
nasm-tc(){  cd $(env-home)/tools/nasm ; }


nasm-env(){      elocal- ; nasm-path ;  }
nasm-path(){  PATH=$LOCAL_BASE/env/bin:$PATH ; }
nasm-info(){ nasm -v ; which nasm ;  }

nasm-get(){
   local dir=$(dirname $(nasm-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(nasm-url) 
   local nam=$(nasm-nam) 
   local dst=$(basename $url)

   [ ! -f $dst ] && curl -L -O $url
   [ ! -d $nam ] && tar zxvf $dst
}

nasm-configure(){

   nasm-cd
   ./configure --prefix=$(local-base)/env

}

nasm--()
{
   nasm-get
   nasm-configure

   make          
   #make everything    # building docs fails on OSX for "cp -ufv" no -u option, and lack of some perl deps
   make install

}

nasm-hello()
{
   nasm-tcd
   nasm -felf64 hello.asm && ld hello.o && ./a.out
   # no C library, just system calls 

}

