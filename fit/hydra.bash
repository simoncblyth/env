hydra-source(){   echo ${BASH_SOURCE} ; }
hydra-vi(){       vi $(hydra-source) ; }
hydra-env(){      elocal- ; }
hydra-usage(){ cat << EOU


* https://github.com/MultithreadCorner/Hydra

* https://media.readthedocs.org/pdf/hydra-documentation/latest/hydra-documentation.pdf
* https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern


EOU
}
hydra-dir(){ echo $(local-base)/env/fit/Hydra ; }
hydra-cd(){  cd $(hydra-dir); }
hydra-get(){
   local dir=$(dirname $(hydra-dir)) &&  mkdir -p $dir && cd $dir
   git clone https://github.com/MultithreadCorner/Hydra
}
