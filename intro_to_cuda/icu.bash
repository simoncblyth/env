# === func-gen- : intro_to_cuda/icu.bash fgp intro_to_cuda/icu.bash fgn icu fgh intro_to_cuda
icu-src(){      echo intro_to_cuda/icu.bash ; }
icu-source(){   echo ${BASH_SOURCE:-$(env-home)/$(icu-src)} ; }
icu-vi(){       vi $(icu-source) ; }
icu-env(){      elocal- ; }
icu-usage(){ cat << EOU

Introduction To CUDA
=======================


CUDA Documentation/Download
-----------------------------

* http://docs.nvidia.com/cuda/index.html

Hello World Examples
----------------------

* https://bitbucket.org/simoncblyth/env/src/tip/intro_to_cuda/
 

GPU Intro
----------

* https://blogs.nvidia.com/blog/2009/12/16/whats-the-difference-between-a-cpu-and-a-gpu/

CUDA Intro
-----------

* http://on-demand.gputechconf.com/gtc-express/2011/presentations/GTC_Express_Sarah_Tariq_June2011.pdf


cudaMalloc : why void** ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    int* ptr = 0;
    void** ptr_to_ptr = &ptr;
    cudaMalloc(ptr_to_ptr, sizeof(int));
    assert(ptr != 0);
    // ptr now points to a segment of device memory


Thrust
----------

* http://on-demand.gputechconf.com/gtc/2012/presentations/S0602-Intro-to-Thrust-Parallel-Algorithms-Library.pdf


GTC Search for CUDA
------------------------

* http://on-demand-gtc.gputechconf.com/gtcnew/on-demand-gtc.php?searchByKeyword=CUDA&searchItems=&sessionTopic=&sessionEvent=&sessionYear=&sessionFormat=&submit=&select=



EOU
}
icu-dir(){ echo $(env-home)/intro_to_cuda ; }
icu-cd(){  cd $(icu-dir); }
icu-c(){   cd $(icu-dir); }
icu-get(){
   local dir=$(dirname $(icu-dir)) &&  mkdir -p $dir && cd $dir

}

icu-refs(){ cd ~/intro_to_cuda_refs ; }
