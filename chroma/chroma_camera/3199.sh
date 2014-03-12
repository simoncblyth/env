#!/bin/bash -l

chroma-

cd $ENV_HOME/chroma/chroma_camera
cuda-gdb --args python -m pycuda.debug simplecamera.py -s3199 -d3 -f10 --eye=0,1,0 --lookat=10,0,10 -G




