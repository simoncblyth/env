tritonbackend-env(){ echo -n ; }
tritonbackend-vi(){  vi $BASH_SOURCE ; }
tritonbackend-usage(){ cat << EOU

I am curious on the high level workflow for developing a custom C++ triton
server backend for the Opticks optical simulation package when the convenient
way to run the server is from a docker container.   Does that mean I must build
the opticks_backend opticks package dependencies inside the docker container ?
I am not familiar with docker based development.


Building them inside the container is the safest way to guarantee binary
compatibility and avoid a nightmare of missing symbols or library version
mismatches




Dockerfile::

    # Start with the official Triton image that matches your targeted release
    FROM nvcr.io/nvidia/tritonserver:26.02-py3

    # Install standard C++ build utilities
    RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        git \
        libssl-dev \
        && rm -rf /var/lib/apt/lists/*

    # [CRITICAL]: Install Opticks core dependencies here
    # (e.g., Geant4, specific header/library copies of NVIDIA OptiX)
    # RUN git clone ... && cmake && make install




docker compose approach
---------------------------

Think of the Dockerfile as the blueprint for building your software
environment, and the docker-compose.yml as the blueprint for running it. Even
for a single container, it saves time and prevents mistakes.




Once the docker-compose.yml file sits in your root project folder, 
your entire development loop becomes incredibly elegant.
To start the sandbox environment::

   docker compose up -d 

It reads the file, handles all the GPU and CVMFS mappings, and runs smoothly in the background.
To jump inside and compile your C++ backend::

    docker compose exec triton-opticks-dev bash

You are instantly dropped into a bash prompt inside the environment 
where you can type make or cmake. To completely shut down and clean up::

    docker compose down

It stops the container and neatly unmounts your paths without losing 
any local source code adjustments.









EOU
}



tritonbackend-image(){ echo my-opticks-triton-dev-image:latest ; }

tritonbackend-run(){

    : cmake building is done inside the container created from the image here 
    docker run --gpus all -it --rm \
      -v /cvmfs:/cvmfs:shared \
      -v /usr/local/triton/opticks_backend_source:/workspace/backend \
      -v /usr/local/triton/model_repository:/opt/tritonserver/model_repository \
      $(tritonbackend-image) bash


}







