rng-source(){   echo ${BASH_SOURCE} ; }
rng-dir(){     echo $(dirname $(rng-source)) ; }
rng-vi(){       vi $(rng-source) ; }
rng-env(){      elocal- ; }
rng-usage(){ cat << EOU

RNG : Simple RNG that can be coded in CUDA 
============================================

* https://en.wikipedia.org/wiki/Middle-square_method

Hmm thinking about finding some simple PRNG that 
can be implemented in CUDA and C++ and gives same sequence
on GPU or CPU : to avoid having to move random buffers of 
around...

But why reinvent the wheel : can interface 
a CUDA RNG algo as an CLHEP::HepRandom engine, 
can use a big buffer



* https://thrust.github.io/doc/group__predefined__random.html

g4-cls Randomize::

     26 // Including Engines ...
     27 
     28 #include "CLHEP/Random/DualRand.h"
     29 #include "CLHEP/Random/JamesRandom.h"
     30 #include "CLHEP/Random/MixMaxRng.h"
     31 #include "CLHEP/Random/MTwistEngine.h"
     32 #include "CLHEP/Random/RanecuEngine.h"
     33 #include "CLHEP/Random/RanluxEngine.h"
     34 #include "CLHEP/Random/Ranlux64Engine.h"
     35 #include "CLHEP/Random/RanshiEngine.h"
     36 






EOU
}
rng-cd(){  cd $(rng-dir); }
rng-c(){  cd $(rng-dir); }
