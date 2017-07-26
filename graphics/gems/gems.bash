# === func-gen- : graphics/gems/gems fgp graphics/gems/gems.bash fgn gems fgh graphics/gems
gems-src(){      echo graphics/gems/gems.bash ; }
gems-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gems-src)} ; }
gems-vi(){       vi $(gems-source) ; }
gems-env(){      elocal- ; }
gems-usage(){ cat << EOU





EOU
}
gems-dir(){ echo $(local-base)/env/graphics/gems/GraphicsGems ; }
gems-c(){   cd $(gems-dir); }
gems-cd(){  cd $(gems-dir); }
gems-mate(){ mate $(gems-dir) ; }
gems-get(){
   local dir=$(dirname $(gems-dir)) &&  mkdir -p $dir && cd $dir


   
   [ ! -d GraphicsGems ] && git clone https://github.com/erich666/GraphicsGems 

}

gems-f(){  gems-c ; find . -type f -exec grep -H ${1:-torus} {} \; ; } 

gems-open(){ open $(gems-dir)/index.html ; }
