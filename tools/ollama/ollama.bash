ollama-vi(){ vi $BASH_SOURCE ; }
ollama-env(){ echo -n ; }
ollama-usage(){ cat << EOU
tools/ollama/ollama.bash
==========================

Linux control of ollama service
--------------------------------

::

     sudo systemctl status ollama
     sudo systemctl start ollama
     sudo systemctl restart ollama


List models
------------

::

    A[blyth@localhost ~]$ ollama list
    NAME                     ID              SIZE      MODIFIED     
    gemma4:26b-a4b-it-qat    2dd70431afed    15 GB     2 days ago      
    gemma4:12b               4eb23ef187e2    7.6 GB    2 days ago      
    gemma4:12b-it-qat        38044be4f923    7.2 GB    2 days ago      
    gemma3:4b                a2af6cc3eb7f    3.3 GB    3 months ago    
    qwen3:8b                 500a1f067a9f    5.2 GB    9 months ago    
    A[blyth@localhost ~]$ 







Ollama installation on Linux
------------------------------


::

    A[blyth@localhost sysrap]$ cat ~/env/tools/ollama/ollama_install.sh | sh 
    >>> Installing ollama to /usr/local
    [sudo] password for blyth: 
    >>> Downloading ollama-linux-amd64.tar.zst
    ######################################################################## 100.0%
    >>> Creating ollama user...
    >>> Adding ollama user to render group...
    >>> Adding ollama user to video group...
    >>> Adding current user to ollama group...
    >>> Creating ollama systemd service...
    >>> Enabling and starting ollama service...
    Created symlink /etc/systemd/system/default.target.wants/ollama.service → /etc/systemd/system/ollama.service.
    >>> NVIDIA GPU installed.



open-webui
-----------

open-webui Starts unhealthy while downloading resources::

    A[blyth@localhost ~]$ docker ps
    CONTAINER ID   IMAGE                                COMMAND           CREATED         STATUS                     PORTS                                         NAMES
    7aa1c80abfef   ghcr.io/open-webui/open-webui:main   "bash start.sh"   2 minutes ago   Up 2 minutes (unhealthy)   0.0.0.0:3000->8080/tcp, [::]:3000->8080/tcp   open-webui

Follow the log of whats happening in the container::

    A[blyth@localhost ~]$ docker logs -f open-webui








EOU
}

ollama-open-webui-old(){

   #docker run -d -p 3000:8080            --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
   docker run -d -p 3000:8080 --gpus all --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:cuda
}

ollama-open-webui-port(){ echo 7860 ; }
ollama-open-webui(){

   : docker rm -f open-webui

   docker run -d \
        --network=host \
        --gpus all \
        -v open-webui:/app/backend/data \
        -e OLLAMA_BASE_URL=http://127.0.0.1:11434 \
        -e PORT=$(ollama-open-webui-port) \
        --name open-webui \
        --restart always \
        ghcr.io/open-webui/open-webui:cuda

   : docker ps
   : docker logs -f open-webui
}

ollama-open-webui-stop()
{
   type $FUNCNAME
   docker stop open-webui
   docker ps
   sleep 1
   sudo systemctl stop ollama
}



ollama-rsync-to-workstation() 
{
    echo "🚀 Starting rsync transfer from Mac to Linux..."
    rsync -avz --progress ~/.ollama/models/ AD:~/ollama_incoming/

    echo -e "\n✅ Transfer complete!"
    echo "On Linux workstation run : ollama-adopt-incoming"
}

ollama-adopt-incoming()
{
   type $FUNCNAME

   # 1. Merge the massive blobs safely (skipping duplicates)
   # --remove-source-files
   sudo rsync -av ~/ollama_incoming/blobs/ /usr/share/ollama/.ollama/models/blobs/

   # 2. Merge the manifests folder safely (skipping duplicates/preserving structure)
   # --remove-source-files
   sudo rsync -av ~/ollama_incoming/manifests/ /usr/share/ollama/.ollama/models/manifests/

   # 3. Fix the ownership to the ollama system user
   sudo chown -R ollama:ollama /usr/share/ollama/.ollama/models/

   # 4. Restart the service
   sudo systemctl restart ollama

   # 5. Clean up any leftover duplicate blobs/folders
   #rm -rf ~/ollama_incoming

}



