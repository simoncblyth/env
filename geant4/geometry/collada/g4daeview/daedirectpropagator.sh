#!/bin/bash -l

usage(){ cat << EOU

DAEDIRECTPROPAGATOR.SH
========================

For debugging, CPL and Photon conversions.
For full picture see g4daechroma.sh::

    g4daeview-
    g4daeview-cd
    ./daedirectpropagator.sh

    (chroma_env)delta:g4daeview blyth$ ./daedirectpropagator.sh 
    2014-09-19 17:23:56,146 env.chroma.ChromaPhotonList.cpl:68  load_cpl from /usr/local/env/tmp/1.root 
    <ROOT.ChromaPhotonList object ("ChromaPhotonList") at 0x7fcba3a08fe0>
    (chroma_env)delta:g4daeview blyth$ 

EOU
}


chroma-

export-
export-export   # DAE_NAME_* and DAE_PATH_TEMPLATE

zmqroot-
zmqroot-export  # ZMQROOT_LIB  

cpl-
cpl-export      # CHROMAPHOTONLIST_LIB


if [ "$NODE_TAG" == "N" ]; then
   #
   # for parasitic use of NuWa python2.7 on N with easy_installed pyzmq, see pyzmq-
   # although no GPU/CUDA/Chroma on N, it would be instructive to have this
   # operational there to some extent to check CPL transport 
   # 
   fenv 
   export ZMQROOT_LIB=$DYB/NuWa-trunk/dybgaudi/InstallArea/$CMTCONFIG/lib/libZMQRoot.so
   export CHROMAPHOTONLIST_LIB=$DYB/NuWa-trunk/dybgaudi/InstallArea/$CMTCONFIG/lib/libChroma.so
   env | grep ZMQ 
   env | grep CHROMA
fi

daedirectpropagator.py $*




