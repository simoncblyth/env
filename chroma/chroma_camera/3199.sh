#!/bin/bash -l

chroma-

cd $ENV_HOME/chroma/chroma_camera
cuda-gdb --args python -m pycuda.debug simplecamera.py -s3199 -d3 -f10 --eye=0,1,0 --lookat=0,0,0 -G -o 3199_000.png




