# === func-gen- : cuda/optix/optix301/sample1/optixsample1 fgp cuda/optix/optix301/sample1/optixsample1.bash fgn optixsample1 fgh cuda/optix/optix301/sample1
optixsample1-src(){      echo cuda/optix/optix301/sample1manual/optixsample1.bash ; }
optixsample1-source(){   echo ${BASH_SOURCE:-$(env-home)/$(optixsample1-src)} ; }
optixsample1-vi(){       vi $(optixsample1-source) ; }
optixsample1-usage(){ cat << EOU





EOU
}

optixsample1-env(){      
   elocal- 
   optix-
   optix-export 
}


optixsample1-dir(){  echo $(env-home)/cuda/optix/optix301/sample1manual ; }
optixsample1-sdir(){ echo "$(env-home)/cuda/optix/optix301/sample1manual" ; }
optixsample1-bdir(){ echo "$(local-base)/env/cuda/optix/optix301/sample1manual" ; }

optixsample1-cd(){   cd $(optixsample1-dir); }
optixsample1-scd(){  cd $(optixsample1-sdir); }
optixsample1-bcd(){  cd $(optixsample1-bdir); }


optixsample1-cmake(){

   local bdir=$(optixsample1-bdir)

   rm -rf $bdir
  
   mkdir -p $bdir

   optixsample1-bcd

   cmake  -DCUDA_NVCC_FLAGS="-ccbin /usr/bin/clang" "$(optixsample1-sdir)"
}

optixsample1-cmake-clean(){

   ## huh cmake is dumping in the sdir ?
   optixsample1-scd
   rm -rf CMakeFiles CMakeCache.txt include
}


