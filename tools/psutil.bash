# === func-gen- : tools/psutil fgp tools/psutil.bash fgn psutil fgh tools
psutil-src(){      echo tools/psutil.bash ; }
psutil-source(){   echo ${BASH_SOURCE:-$(env-home)/$(psutil-src)} ; }
psutil-vi(){       vi $(psutil-source) ; }
psutil-env(){      elocal- ; }
psutil-usage(){ cat << EOU

psutil : A cross-platform process and system utilities module for Python
===========================================================================

http://code.google.com/p/psutil/

psutil is a module providing an interface for retrieving information on all
running processes and system utilization (CPU, memory, disks, network, users)
in a portable way by using Python, implementing many functionalities offered by
command line tools such as::

    ps
    top
    df
    kill
    free
    lsof
    netstat
    ifconfig
    nice
    ionice
    iostat
    iotop
    uptime
    pidof
    tty
    who
    taskset
    pmap


EOU
}
psutil-dir(){ echo $(local-base)/env/tools/psutil ; }
psutil-cd(){  cd $(psutil-dir); }
psutil-mate(){ mate $(psutil-dir) ; }
psutil-get(){
   local dir=$(dirname $(psutil-dir)) &&  mkdir -p $dir && cd $dir

   hg clone https://code.google.com/p/psutil/

}
