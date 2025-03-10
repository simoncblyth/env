sysrap-src(){      echo sysrap/sysrap.bash ; }
sysrap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sysrap-src)} ; }
sysrap-vi(){       vi $(sysrap-source) ; }
sysrap-usage(){ cat << \EOU

System Rap
============

Lowest level package, beneath BoostRap and 
**explicitly not using Boost**. 

A lower level pkg that BoostRap is required 
as nvcc (CUDA compiler) has trouble compiling 
Boost headers.

EOU
}

sysrap-env(){      elocal- ; opticks- ;  }

sysrap-dir(){  echo $(sysrap-sdir) ; }
sysrap-sdir(){ echo $(env-home)/sysrap ; }
sysrap-tdir(){ echo $(env-home)/sysrap/tests ; }
sysrap-idir(){ echo $(opticks-idir); }
sysrap-bdir(){ echo $(opticks-bdir)/sysrap ; }

sysrap-cd(){   cd $(sysrap-sdir); }
sysrap-scd(){  cd $(sysrap-sdir); }
sysrap-tcd(){  cd $(sysrap-tdir); }
sysrap-icd(){  cd $(sysrap-idir); }
sysrap-bcd(){  cd $(sysrap-bdir); }

sysrap-name(){ echo SysRap ; }
sysrap-tag(){  echo SYSRAP ; }

sysrap-wipe(){    local bdir=$(sysrap-bdir) ; rm -rf $bdir ; }

sysrap--(){       opticks-- $(sysrap-bdir) ; } 
sysrap-ctest(){   opticks-ctest $(sysrap-bdir) $* ; } 
sysrap-genproj(){ sysrap-scd ; opticks-genproj $(sysrap-name) $(sysrap-tag) ; } 
sysrap-gentest(){ sysrap-tcd ; opticks-gentest ${1:-SCheck} $(sysrap-tag) ; } 
sysrap-txt(){     vi $(sysrap-sdir)/CMakeLists.txt $(sysrap-tdir)/CMakeLists.txt ; } 


