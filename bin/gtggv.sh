#!/bin/bash -l

source /afs/ihep.ac.cn/soft/juno/JUNO-ALL-SLC6/GPU/20150723/bashrc

cmdline="$*"
ggeoview-

if [ "${cmdline/--juno}" != "${cmdline}" ]; then
   export GGEOVIEW_DETECTOR=DAE_NAME_JUNO
elif [ "${cmdline/--dyb}" != "${cmdline}" ]; then
   export GGEOVIEW_DETECTOR=DAE_NAME_DYB
else
   export GGEOVIEW_DETECTOR=DAE_NAME_JUNO
fi

#ggeoview-vrun $*
#export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0
ggeoview-gdb $* --size 800,600,1


