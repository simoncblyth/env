# === func-gen- : tools/docker fgp tools/docker.bash fgn docker fgh tools
docker-src(){      echo tools/docker.bash ; }
docker-source(){   echo ${BASH_SOURCE:-$(env-home)/$(docker-src)} ; }
docker-vi(){       vi $(docker-source) ; }
docker-env(){      elocal- ; }
docker-usage(){ cat << EOU

Docker
=========


GPU containerization ?
-------------------------

* https://stackoverflow.com/questions/25185405/using-gpu-from-a-docker-container
* http://www.nvidia.com/object/docker-container.html
* https://github.com/NVIDIA/nvidia-docker
* http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements



* https://github.com/NVIDIA/nvidia-docker/wiki/Motivation

  The GPU driver lives on the host, not in the container


* https://docs.docker.com/engine/reference/run/#general-form

* https://www.youtube.com/watch?v=YFl2mCHdv24

  12 min intro 



EOU
}
docker-dir(){ echo $(local-base)/env/tools/tools-docker ; }
docker-cd(){  cd $(docker-dir); }
docker-mate(){ mate $(docker-dir) ; }
docker-get(){
   local dir=$(dirname $(docker-dir)) &&  mkdir -p $dir && cd $dir

}
