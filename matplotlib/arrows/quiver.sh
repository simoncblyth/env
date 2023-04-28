#!/bin/bash -l 

name=quiver
DIR=$(dirname $BASH_SOURCE)

${IPYTHON:-ipython} --pdb -i $DIR/$name.py 



