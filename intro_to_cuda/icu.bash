# === func-gen- : intro_to_cuda/icu.bash fgp intro_to_cuda/icu.bash fgn icu fgh intro_to_cuda
icu-src(){      echo intro_to_cuda/icu.bash ; }
icu-source(){   echo ${BASH_SOURCE:-$(env-home)/$(icu-src)} ; }
icu-vi(){       vi $(icu-source) ; }
icu-env(){      elocal- ; }
icu-usage(){ cat << EOU

Introduction To CUDA
=======================



CUDA Intro
-----------

* http://www.nvidia.com/content/GTC-2010/pdfs/2131_GTC2010.pdf

* http://on-demand.gputechconf.com/gtc-express/2011/presentations/GTC_Express_Sarah_Tariq_June2011.pdf


Thrust
----------

* http://on-demand.gputechconf.com/gtc/2012/presentations/S0602-Intro-to-Thrust-Parallel-Algorithms-Library.pdf

* http://ppomorsk.sharcnet.ca/CSE746/lecture8_CSE746_2014.pdf





EOU
}
icu-dir(){ echo $(env-home)/intro_to_cuda ; }
icu-cd(){  cd $(icu-dir); }
icu-c(){   cd $(icu-dir); }
icu-get(){
   local dir=$(dirname $(icu-dir)) &&  mkdir -p $dir && cd $dir

}

icu-refs(){ cd ~/intro_to_cuda_refs ; }
