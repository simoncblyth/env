#!/bin/bash 
usage(){ cat << EOU
pipeline.sh
============

~/env/graphics/pipeline/pipeline.sh


EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

script=pipeline.py 

${IPYTHON:-ipython} -i --pdb $script
