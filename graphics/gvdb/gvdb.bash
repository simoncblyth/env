# === func-gen- : graphics/gvdb/gvdb fgp graphics/gvdb/gvdb.bash fgn gvdb fgh graphics/gvdb
gvdb-src(){      echo graphics/gvdb/gvdb.bash ; }
gvdb-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gvdb-src)} ; }
gvdb-vi(){       vi $(gvdb-source) ; }
gvdb-env(){      elocal- ; }
gvdb-usage(){ cat << EOU

NVIDIA GVDB Voxels
====================

Sparse volume compute and rendering on NVIDIA GPUs

* https://developer.nvidia.com/gvdb
* https://github.com/NVIDIA/gvdb-voxels

* http://www.ramakarl.com/website/wp-content/uploads/GVDB_HPG2016_CRC.pdf
* ~/opticks_refs/GVDB_HPG2016_CRC.pdf


NVIDIA GVDB Voxels was released as an SDK with samples at the GPU Technology
Conference in 2017 with an open source license (BSD 3-clause) enabling
developers the greatest flexibility in creating novel applications for NVIDIA
CUDA-based GPUs.



EOU
}
gvdb-dir(){ echo $(local-base)/env/graphics/gvdb/graphics/gvdb-gvdb ; }
gvdb-cd(){  cd $(gvdb-dir); }
gvdb-mate(){ mate $(gvdb-dir) ; }
gvdb-get(){
   local dir=$(dirname $(gvdb-dir)) &&  mkdir -p $dir && cd $dir

}
