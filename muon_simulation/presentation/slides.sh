#!/bin/bash -l 

slides-

# override the generic functions
slides-name(){      echo ${SLIDES_NAME:-nov2013_gpu_nuwa} ; }
slides-branch(){    echo ${SLIDES_BRANCH:-muon_simulation/presentation} ; }        # env relative path to where .txt sources reside
slides-host(){      echo ${SLIDES_HOST:-dayabay.phys.ntu.edu.tw} ; }   

# this assumes the S5 slides are already created and available at the URL 
# the above parameters correspond to 

slides-get 0 17

