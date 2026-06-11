# === func-gen- : tools/docker fgp tools/docker.bash fgn docker fgh tools
docker-src(){      echo tools/docker.bash ; }
docker-source(){   echo ${BASH_SOURCE:-$(env-home)/$(docker-src)} ; }
docker-vi(){       vi $(docker-source) ; }
docker-env(){      elocal- ; }
docker-usage(){ cat << EOU

Docker
=========


nvidia-docker
-------------

* https://github.com/NVIDIA/nvidia-docker


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


get docker to pull using socks proxy
-------------------------------------

::

    A[blyth@localhost ~]$ sudo cat /etc/systemd/system/docker.service.d/http-proxy.conf
    [Service]
    Environment="HTTP_PROXY=socks5://127.0.0.1:8080"
    Environment="HTTPS_PROXY=socks5://127.0.0.1:8080"
    Environment="NO_PROXY=localhost,127.0.0.1"


    sudo systemctl daemon-reload
    sudo systemctl restart docker

    docker pull alpine


maybe 
---------

::

    A[blyth@localhost ~]$ sudo cat /etc/systemd/system/docker.service.d/http-proxy.conf
    [Service]
    Environment="HTTP_PROXY=socks5h://172.17.0.1:8080"
    Environment="HTTPS_PROXY=socks5h://172.17.0.1:8080"
    Environment="NO_PROXY=localhost,127.0.0.1"



(GEMINI) network contexts relevant to docker over socks proxy
----------------------------------------------------------------

::

    [ Your Host Machine ] <--- (SOCKS Proxy Listening here)
      │
      ├── Context 1: The Docker Daemon (Systemd) 
      │     └── Pulls images, talks to Docker Hub. Sees host as "127.0.0.1".
      │
      ├── Context 2: The Build Sandbox (BuildKit)
      │     └── Handles "FROM" image lookups. Lives in an isolated client/daemon bridge.
              It expects configurations passed from the user's local ~/.docker/config.json file, 
              using either the default bridge gateway or host.docker.internal.


      │
      └── Context 3: The Container Environment (RUN/Execution)
            └── Ephemeral sandbox running "dnf/yum/apt". Sees host as "172.17.0.1".

             The IP to use: 172.17.0.1 (or host.docker.internal). 
             To a container, 127.0.0.1 means itself. 
             It cannot see your host's localhost. 
             It must talk to the host via the network gateway.


If you want a flawless proxy setup across all contexts, configure them like this:

For docker pull / Daemon:

* Use socks5h://127.0.0.1:8080 in /etc/systemd/system/docker.service.d/http-proxy.conf.

For docker build (FROM metadata lookups):

* Use socks5h://172.17.0.1:8080 in your user's ~/.docker/config.json.

For Package Managers inside Containers (dnf/apt):

* Pass --build-arg HTTP_PROXY="socks5h://172.17.0.1:8080" into your build command.

⚠️ Crucial Security Note: Whenever you route containers to 172.17.0.1:8080, your
SOCKS proxy on the host must be listening on 0.0.0.0 (all interfaces) rather
than just 127.0.0.1, otherwise it will drop the incoming connection from the
Docker bridge network.



BuildKit config
---------------

~/.docker/config.json::

    {
      "proxies": {
        "default": {
          "httpProxy": "socks5h://172.17.0.1:8080",
          "httpsProxy": "socks5h://172.17.0.1:8080",
          "noProxy": "localhost,127.0.0.1"
        }
      }
    }




EOU
}
docker-dir(){ echo $(local-base)/env/tools/tools-docker ; }
docker-cd(){  cd $(docker-dir); }
docker-mate(){ mate $(docker-dir) ; }
docker-get(){
   local dir=$(dirname $(docker-dir)) &&  mkdir -p $dir && cd $dir

}
