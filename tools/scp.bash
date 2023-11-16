scp-vi(){   vi $BASH_SOURCE ; }
scp-env(){  echo -n ; }
scp-usage(){ cat << EOU
scp
===

recursively copy contents of remote directory to this one
----------------------------------------------------------

* use the "." to mean the contents of the directory and not the directory itself

::

    epsilon:opticks_download_cache blyth$ pwd
    /data/opticks_download_cache
    epsilon:opticks_download_cache blyth$ scp -r P:/data/opticks_download_cache/. .
    ##                                                                          ^

EOU
}
