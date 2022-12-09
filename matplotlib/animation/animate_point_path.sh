#!/bin/bash -l 

name=$(basename $BASH_SOURCE)
name=${name/.sh}
echo $name

${IPYTHON:-ipython} --pdb -i $name.py 


