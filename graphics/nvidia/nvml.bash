nvml-source(){   echo ${BASH_SOURCE} ; }
nvml-edir(){ echo $(dirname $(nvml-source)) ; }
nvml-ecd(){  cd $(nvml-edir); }
nvml-dir(){  echo $LOCAL_BASE/env/graphics/nvidia/nvml/example ; }
nvml-cd(){  cd $(nvml-dir) ; }
nvml-vi(){   vi $(nvml-source) ; }
nvml-env(){  elocal- ; }
nvml-usage(){ cat << EOU

NVML : programmatic nvidia-smi
===================================

::

    [blyth@localhost example]$ ./example 
    Found 2 devices

    Listing devices:
    0. TITAN RTX [00000000:73:00.0]
         Changing device's compute mode from 'Default' to 'Prohibited'
             Need root privileges to do that: Insufficient Permissions
    1. TITAN V [00000000:A6:00.0]
         Changing device's compute mode from 'Default' to 'Prohibited'
             Need root privileges to do that: Insufficient Permissions
    All done.
    Press ENTER to continue...


EOU
}

nvml-info(){ cat << EOI

   nvml-dir : $(nvml-dir)

EOI
}

nvml-get(){
    local dir=$(dirname $(nvml-dir)) &&  mkdir -p $dir && cd $dir
    if [ -d "example" ]; then 
        echo $msg example exists already 
    else
        local cmd="cp -R $(nvml-cudadir)/nvml/example ."
        echo $msg $cmd
        eval $cmd
    fi 
}

nvml-cudadir(){ echo /usr/local/cuda-10.1 ; } 

nvml-make()
{
    nvml-cd
    CFLAGS="-I$(nvml-cudadir)/targets/x86_64-linux/include" make -e    # -e option allows to override settings inside Makefile with envvars
    ./example
}

nvml--(){
    nvml-get 
    nvml-make 
}




