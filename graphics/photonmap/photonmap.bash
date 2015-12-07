# === func-gen- : graphics/photonmap/photonmap fgp graphics/photonmap/photonmap.bash fgn photonmap fgh graphics/photonmap
photonmap-src(){      echo graphics/photonmap/photonmap.bash ; }
photonmap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(photonmap-src)} ; }
photonmap-vi(){       vi $(photonmap-source) ; }
photonmap-env(){      elocal- ; }
photonmap-usage(){ cat << EOU

https://code.google.com/p/275a-photonmap/source/checkout


https://www.inf.ed.ac.uk/publications/thesis/online/IM090675.pdf

Photon Mapping on the GPU
Martin Fleisz

http://www.ks.uiuc.edu/Research/vmd/doxygen/CUDASpatialSearch_8cu-source.html



http://stackoverflow.com/questions/29445429/how-to-bring-equal-elements-together-using-thrust-without-sort

https://sites.google.com/a/compgeom.com/stann/


http://http.developer.nvidia.com/GPUGems3/gpugems3_ch32.html


Spatial Data Structures, Sorting and GPU Parallelism for Situated-agent Simulation and Visualisation

http://worldcomp-proceedings.com/proc/p2012/MSV2429.pdf



Fast Uniform Grid Construction on GPGPUs Using Atomic Operations

http://www.ce.uniroma2.it/publications/parco2013_uniformgrids.pdf




EOU
}
photonmap-dir(){ echo $(local-base)/env/graphics/photonmap ; }
photonmap-cd(){  cd $(photonmap-dir); }
photonmap-mate(){ mate $(photonmap-dir) ; }
photonmap-get(){
   local dir=$(dirname $(photonmap-dir)) &&  mkdir -p $dir && cd $dir

   svn checkout http://275a-photonmap.googlecode.com/svn/trunk/ photonmap
}
