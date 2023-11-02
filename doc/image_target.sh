#!/bin/bash -l 

name=image_target
path=/tmp/$name.jpg
export IMAGE_TARGET_PATH=$path

${IPYTHON:-ipython} --pdb  $name.py 

open $path


