
triton-env(){  echo -n ; }
triton-vi(){  vi $BASH_SOURCE ; }
triton-usage(){ cat << EOU
Getting Familiar with NVIDIA Triton Inference Server
======================================================


Top level
----------

* https://github.com/triton-inference-server



Random Walk thru Triton docs
------------------------------

https://github.com/triton-inference-server/server

https://github.com/triton-inference-server/backend/blob/main/README.md#triton-backend-api

https://arxiv.org/pdf/2312.06838
    Optimizing High Throughput Inference on Graph
    Neural Networks at Shared Computing Facilities
    with the NVIDIA Triton Inference Server

https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html

https://github.com/triton-inference-server/tutorials

https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#runcont

https://github.com/triton-inference-server/tutorials/blob/main/Quick_Deploy/vLLM/README.md

https://github.com/triton-inference-server/vllm_backend/tree/main


tritonserver images catalog
-----------------------------

* https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver?version=26.04-vllm-python-py3



server
------

* https://github.com/triton-inference-server/server

* https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/compose.md#customize-triton-container

* https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/compose.md#build-it-yourself

Build it yourself
If you would like to do what compose.py is doing under the hood yourself, you
can run compose.py with the --dry-run option and then modify the
Dockerfile.compose file to satisfy your needs.

* https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/compose.md#triton-with-unsupported-and-custom-backends

You can create and build your own Triton backend. The result of that build
should be a directory containing your backend shared library and any additional
files required by the backend. Assuming your backend is called "mybackend" and
that the directory is "./mybackend", adding the following to the Dockerfile
compose.py created will create a Triton image that contains all the supported
Triton backends plus your custom backend.

COPY ./mybackend /opt/tritonserver/backends/mybackend

You also need to install any additional dependencies required by your backend
as part of the Dockerfile. Then use Docker to create the image.

$ docker build -t tritonserver_custom -f Dockerfile.compose .




backend : "create and build your own Triton backend"
---------------------------------------------------------

* https://github.com/triton-inference-server/backend

* https://github.com/triton-inference-server/backend#backend-shared-library




vllm_backend
--------------

* https://github.com/triton-inference-server/vllm_backend/tree/main




Installing the vLLM Backend : just try pulling docker image from NGC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A::

    docker pull nvcr.io/nvidia/tritonserver:26.04-vllm-python-py3


take a look around inside the container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A::

    docker run --rm -it --entrypoint /bin/bash nvcr.io/nvidia/tritonserver:26.04-vllm-python-py3   ## triton-bash

    root@29058340062f:/opt/tritonserver# ls -alst
    total 2952
       0 drwxr-xr-x. 1 root          root               56 Apr 24 00:58 .
       0 drwxr-xr-x. 1 root          root               26 Apr 24 00:58 ..
       0 drwxrwxrwx. 4 triton-server triton-server      32 Apr 24 00:56 caches
       0 drwxrwxrwx. 3 triton-server triton-server      22 Apr 24 00:55 repoagents
       0 drwxrwxrwx. 4 triton-server triton-server      32 Apr 24 00:55 backends
       0 drwxrwxrwx. 2 triton-server triton-server      38 Apr 24 00:55 third-party-src
       4 -rw-rw-rw-. 1 triton-server triton-server    1490 Apr 24 00:55 LICENSE
       4 -rw-rw-rw-. 1 triton-server triton-server       7 Apr 24 00:55 TRITON_VERSION
       0 drwxrwxrwx. 2 triton-server triton-server      26 Apr 24 00:55 bin
       0 drwxrwxrwx. 3 triton-server triton-server      20 Apr 24 00:55 include
       0 drwxrwxrwx. 2 triton-server triton-server      32 Apr 24 00:55 lib
       0 drwxrwxrwx. 3 triton-server triton-server     109 Apr 24 00:55 python
    2944 -rw-r--r--. 1 triton-server triton-server 3012640 Apr 24 00:45 NVIDIA_Deep_Learning_Container_License.pdf
    root@29058340062f:/opt/tritonserver#


clone vllm_backend repo
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    git clone https://github.com/triton-inference-server/vllm_backend.git


using socks proxy from inside docker container : to download the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



::

    pip install pysocks
    export HTTP_PROXY="socks5h://172.17.0.1:8080" HTTPS_PROXY="socks5h://172.17.0.1:8080" http_proxy="socks5h://172.17.0.1:8080" https_proxy="socks5h://172.17.0.1:8080"
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='facebook/opt-125m', local_dir='./model_cache/opt-125m')
"


On first try that failed. Had to change the socks proxy from localhost:8080 to 0.0.0.0:8080

* because triton-bash has /work inside container bound to ~/vllm_backend from host - the download will not be lost on exiting the container


Manual model_cache downloaded
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    A[blyth@localhost vllm_backend]$ l model_cache/opt-125m/
    total 735496
         4 drwxr-xr-x. 3 root root      4096 May 20 19:24 .
         0 drwxr-xr-x. 3 root root        22 May 20 19:24 ..
    244672 -rw-r--r--. 1 root root 250540281 May 20 19:24 pytorch_model.bin
    244836 -rw-r--r--. 1 root root 250709016 May 20 19:23 tf_model.h5
    244616 -rw-r--r--. 1 root root 250485441 May 20 19:22 flax_model.msgpack
       880 -rw-r--r--. 1 root root    898822 May 20 19:20 vocab.json
         4 -rw-r--r--. 1 root root       685 May 20 19:20 tokenizer_config.json
         4 -rw-r--r--. 1 root root       441 May 20 19:20 special_tokens_map.json
         8 -rw-r--r--. 1 root root      7099 May 20 19:20 README.md
       448 -rw-r--r--. 1 root root    456318 May 20 19:20 merges.txt
         4 -rw-r--r--. 1 root root       137 May 20 19:20 generation_config.json
        12 -rw-r--r--. 1 root root     11117 May 20 19:20 LICENSE.md
         4 -rw-r--r--. 1 root root       651 May 20 19:20 config.json
         4 -rw-r--r--. 1 root root      1173 May 20 19:20 .gitattributes
         0 drwxr-xr-x. 3 root root        25 May 20 19:20 .cache


Configure to find the model from  model_cache not some huggingface reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    A[blyth@localhost vllm_backend]$ git diff samples/model_repository/vllm_model/1/model.json
    diff --git a/samples/model_repository/vllm_model/1/model.json b/samples/model_repository/vllm_model/1/model.json
    index 657953c..b1c877e 100644
    --- a/samples/model_repository/vllm_model/1/model.json
    +++ b/samples/model_repository/vllm_model/1/model.json
    @@ -1,5 +1,5 @@
     {
    -    "model":"facebook/opt-125m",
    +    "model": "/work/model_cache/opt-125m",
         "gpu_memory_utilization": 0.1,
         "enforce_eager": true
     }


ip address wierdness
~~~~~~~~~~~~~~~~~~~~~~~

The ip addresses seem peculiar. From inside the container need to refer to the
host with 172.17.0.1 but from outside need to refer to the server running in
the container with 127.0.0.1 ?

From Outside to Inside: Why 127.0.0.1 works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In your triton-run function, you passed the flag: --net=host
This tells Docker not to create an isolated network namespace for the
container. Instead, the Triton process binds directly to your host machine's
actual network interfaces.

From Inside to Outside: Why 172.17.0.1 works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Even though your running container shares the host network via --net=host,
Docker still creates a default virtual bridge interface on your machine named
docker0 when the Docker service starts up.
By default, the host machine assigns itself the IP address 172.17.0.1 on that
docker0 bridge to act as the gateway for any other standard containers running
on the system.

When you want to explicitly target services running on the host from inside a
container environment, 172.17.0.1 is the dependable shortcut route provided by
the kernel to hop out of container logic back onto the host loop.


EOU

}

triton-vllm-bash0(){
   docker run --rm -it --entrypoint /bin/bash $(triton-vllm-image) ;
}
triton-vllm-bash(){
   docker run --net=host --rm -it -v ~/vllm_backend:/work -w /work  $(triton-vllm-image) /bin/bash
}


triton-docker-ip-show(){ ip addr show docker0 ; }
triton-docker-ip(){      echo 172.17.0.1 ; }


triton-vllm-pull(){  docker pull $(triton-vllm-image) ; }
triton-vllm-image(){ echo nvcr.io/nvidia/tritonserver:26.04-vllm-python-py3 ; }

triton-vllm-run(){

   cd ~/vllm_backend

    : --net=host -p 8001:8001
    : net and p dont make sense together the p is ignored and causes warning

   docker run --gpus all -it --rm \
       --net=host \
       --shm-size=1G \
       --ulimit memlock=-1 \
       --ulimit stack=67108864 \
       -v ${PWD}:/work \
       -w /work \
       $(triton-vllm-image) \
       tritonserver \
       --model-repository ./samples/model_repository

}

triton-vllm-query-0()
{
    : nope because used --net=host so need to directly use 127.0.0.1
    curl -X POST $(triton-docker-ip):8000/v2/models/vllm_model/generate -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0}}'
}

triton-vllm-query-1()
{
    curl --noproxy "*" -X POST http://127.0.0.1:8000/v2/models/vllm_model/generate \
         -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0}}'
}

triton-vllm-query-2()
{
    curl --noproxy "*" -X POST http://127.0.0.1:8000/v2/models/vllm_model/generate \
         -d '{"text_input": "What is Triton Inference Server?", "parameters": {"stream": false, "temperature": 0}}'
}

triton-vllm-query()
{
    if [ -z "$1" ]; then
        echo "Error: Please provide a question."
        echo "Usage: triton-query \"Your question here\""
        return 1
    fi

    curl --noproxy "*" -X POST http://127.0.0.1:8000/v2/models/vllm_model/generate \
         -H "Content-Type: application/json" \
         -d '{"text_input": "'"$1"'", "parameters": {"stream": false, "temperature": 0}}'
}


