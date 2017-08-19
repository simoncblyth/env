# === func-gen- : graphics/optix_advanced_samples/oas fgp graphics/optix_advanced_samples/oas.bash fgn oas fgh graphics/optix_advanced_samples
oas-src(){      echo graphics/optix_advanced_samples/oas.bash ; }
oas-source(){   echo ${BASH_SOURCE:-$(env-home)/$(oas-src)} ; }
oas-vi(){       vi $(oas-source) ; }
oas-env(){      elocal- ; }
oas-usage(){ cat << EOU





EOU
}
oas-dir(){ echo $(local-base)/env/graphics/oas/optix_advanced_samples ; }
oas-cd(){  cd $(oas-dir); }
oas-c(){   cd $(oas-dir); }

oas-get(){
   local dir=$(dirname $(oas-dir)) &&  mkdir -p $dir && cd $dir

   local url=https://github.com/nvpro-samples/optix_advanced_samples

   [ ! -d $(basename $url) ] && git clone $url

}
