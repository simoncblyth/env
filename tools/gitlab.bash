gitlab-env(){ echo -n ; } 
gitlab-vi(){ vi $BASH_SOURCE ; }
gitlab-usage(){ cat << EOU



gitlab clone blocked
----------------------

::

    A[blyth@localhost ~]$ HTTP_PROXY=socks5://127.0.0.1:8080 HTTPS_PROXY=socks5://127.0.0.1:8080 git clone https://gitlab.com/nvidia/container-images/cuda.git
    Cloning into 'cuda'...
    fatal: unable to access 'https://gitlab.com/nvidia/container-images/cuda.git/': connection to proxy closed
    A[blyth@localhost ~]$ 

    git -c "http.proxy=socks5h://127.0.0.1:8080"  clone https://gitlab.com/nvidia/container-images/cuda.git 

::

    epsilon:~ blyth$ git -c "http.proxy=socks5h://127.0.0.1:8080"  clone https://gitlab.com/nvidia/container-images/cuda.git 
    Cloning into 'cuda'...
    remote: Enumerating objects: 25221, done.
    remote: Counting objects: 100% (5753/5753), done.
    remote: Compressing objects: 100% (2753/2753), done.
    remote: Total 25221 (delta 2288), reused 5689 (delta 2224), pack-reused 19468 (from 1)
    Receiving objects: 100% (25221/25221), 3.92 MiB | 1.09 MiB/s, done.
    Resolving deltas: 100% (10435/10435), done.
    epsilon:~ blyth$ 

    epsilon:tmp blyth$ git config --global http.proxy
    socks5://127.0.0.1:8080
    epsilon:tmp blyth$ git config --global http.proxy socks5h://127.0.0.1:8080
    epsilon:tmp blyth$ git clone https://gitlab.com/nvidia/container-images/cuda.git
    Cloning into 'cuda'...
    remote: Enumerating objects: 25221, done.
    remote: Counting objects: 100% (5753/5753), done.
    remote: Compressing objects: 100% (2753/2753), done.
    remote: Total 25221 (delta 2288), reused 5689 (delta 2224), pack-reused 19468 (from 1)
    Receiving objects: 100% (25221/25221), 3.92 MiB | 1009.00 KiB/s, done.
    Resolving deltas: 100% (10435/10435), done.





EOU
}
