#!/bin/bash -l

rm -rf build
mkdir build 
cd build 
cmake .. && make 
