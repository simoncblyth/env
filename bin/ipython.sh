#!/bin/bash -l

chroma-
#ipython $* -i 
ipro="g4dae"
if [ -d ~/.ipython/profile_$ipro ]; then 
    ipython --profile=$ipro $*
else
    ipython $*  
fi



