#!/bin/bash -l

#chroma-

export-
export-export

mocknuwa-
mocknuwa-export

env | grep SQLITE3

#ipython $* -i 
#ipro="g4dae"
ipro="g4opticks"

if [ -d ~/.ipython/profile_$ipro ]; then 
    ipython --profile=$ipro $*
else
    ipython $*  
fi



