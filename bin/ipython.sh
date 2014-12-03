#!/bin/bash -l

chroma-

mocknuwa-
mocknuwa-export

#ipython $* -i 
ipro="g4dae"
if [ -d ~/.ipython/profile_$ipro ]; then 
    ipython --profile=$ipro $*
else
    ipython $*  
fi



