#!/bin/bash -l


usage(){ cat << EOU
::

    ./render_pbo.sh --threads-per-block 256 --kernel=render_pbo --max-blocks 512 --view A

EOU
}


chroma-

# cannot profile and memcheck at same time
export CUDA_PROFILE=1  

#cuda-memcheck --force-blocking-launches yes $(which python) render_pbo.py --allsync
$(which python) render_pbo.py  $*

log=cuda_profile_0.log

if [ "$CUDA_PROFILE" == "1" ]; then 
   tail -20 $log
   ls -l $log
fi 

