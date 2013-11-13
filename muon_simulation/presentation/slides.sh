#!/bin/bash -l 
slides-

# override the generic functions
slides-url(){   echo http://dayabay.phys.ntu.edu.tw/e/muon_simulation/presentation/nov2013_gpu_nuwa.html ; }
slides-rdir(){  echo muon_simulation/presentation ; }

# this assumes the S5 slides are already created and available at the above URL
slides-get 0 2

