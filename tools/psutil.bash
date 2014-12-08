# === func-gen- : tools/psutil fgp tools/psutil.bash fgn psutil fgh tools
psutil-src(){      echo tools/psutil.bash ; }
psutil-source(){   echo ${BASH_SOURCE:-$(env-home)/$(psutil-src)} ; }
psutil-vi(){       vi $(psutil-source) ; }
psutil-env(){      elocal- ; }
psutil-usage(){ cat << EOU

psutil : A cross-platform process and system utilities module for Python
===========================================================================

https://github.com/giampaolo/psutil

Formerly http://code.google.com/p/psutil/


psutil is a module providing an interface for retrieving information on all
running processes and system utilization (CPU, memory, disks, network, users)
in a portable way by using Python, implementing many functionalities offered by
command line tools such as ps, top,...



OSX VSIZE 30G for g4daechroma.py 
-----------------------------------

* http://marc-abramowitz.com/archives/2011/12/07/dont-be-alarmed-by-super-large-vsize-on-os-x/






Memory Info
-----------

::

    In [1]: import psutil 

    In [2]: p = psutil.Process()

    In [6]: m = p.get_memory_info()

    In [7]: m
    Out[7]: pmem(rss=49295360L, vms=2597711872L)

    In [8]: m.
    m.count  m.index  m.rss    m.vms    

    In [10]: m.vms/1024/1024
    Out[10]: 2477L

    In [13]: m.rss/1024/1024
    Out[13]: 47L



Installs: G chroma virtual python  2014/12/8
-----------------------------------------------

(chroma_env)delta:psutil blyth$ python setup.py install 
...
Processing psutil-2.2.0-py2.7-macosx-10.9-x86_64.egg
Copying psutil-2.2.0-py2.7-macosx-10.9-x86_64.egg to /usr/local/env/chroma_env/lib/python2.7/site-packages
Adding psutil 2.2.0 to easy-install.pth file

Installed /usr/local/env/chroma_env/lib/python2.7/site-packages/psutil-2.2.0-py2.7-macosx-10.9-x86_64.egg
Processing dependencies for psutil==2.2.0
Finished processing dependencies for psutil==2.2.0



EOU
}
psutil-dir(){ echo $(local-base)/env/tools/psutil ; }
psutil-cd(){  cd $(psutil-dir); }
psutil-mate(){ mate $(psutil-dir) ; }
psutil-get(){
   local dir=$(dirname $(psutil-dir)) &&  mkdir -p $dir && cd $dir

   #hg clone https://code.google.com/p/psutil/
   git clone https://github.com/giampaolo/psutil.git

}
