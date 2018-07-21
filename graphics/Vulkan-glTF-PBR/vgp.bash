# === func-gen- : graphics/Vulkan-glTF-PBR/vgp fgp graphics/Vulkan-glTF-PBR/vgp.bash fgn vgp fgh graphics/Vulkan-glTF-PBR
vgp-src(){      echo graphics/Vulkan-glTF-PBR/vgp.bash ; }
vgp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vgp-src)} ; }
vgp-vi(){       vi $(vgp-source) ; }
vgp-env(){      elocal- ; }
vgp-usage(){ cat << EOU

https://github.com/SaschaWillems/Vulkan-glTF-PBR

unsupported GCC version - see https://github.com/nlohmann/json#supported-compilers


EOU
}
vgp-dir(){ echo $(local-base)/env/graphics/Vulkan-glTF-PBR ; }

vgp-sdir(){ echo $(vgp-dir) ; }
vgp-bdir(){ echo $(vgp-dir).build ; }
vgp-idir(){ echo $(vgp-dir).install ; }

vgp-cd(){  cd $(vgp-dir); }
vgp-bcd(){ cd $(vgp-bdir); }


vgp-get(){
   local dir=$(dirname $(vgp-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d Vulkan-glTF-PBR ] && git clone --recursive https://github.com/SaschaWillems/Vulkan-glTF-PBR.git 
}


vgp-cmake(){
   local bdir=$(vgp-bdir)
   local sdir=$(vgp-sdir)
   mkdir -p $bdir

   cd $bdir
   cmake \
        -DCMAKE_INSTALL_PREFIX=$(vgp-idir) \
        -DJSON_SKIP_UNSUPPORTED_COMPILER_CHECK=on \
           $sdir 


}

vgp-install()
{
   vgp-bcd
   make
}


vgp-run()
{
    vgp-bcd
    cd bin
    LD_LIBRARY_PATH=$(vgp-dir)/libs/vulkan $(vgp-bdir)/bin/Vulkan-glTF-PBR $*
}




