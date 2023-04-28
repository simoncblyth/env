#!/bin/bash -l 

name=arrows
DIR=$(dirname $BASH_SOURCE)

${IPYTHON:-ipython} --pdb -i $DIR/$name.py 



