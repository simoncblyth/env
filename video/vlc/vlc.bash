# === func-gen- : video/vlc/vlc fgp video/vlc/vlc.bash fgn vlc fgh video/vlc
vlc-src(){      echo video/vlc/vlc.bash ; }
vlc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vlc-src)} ; }
vlc-vi(){       vi $(vlc-source) ; }
vlc-env(){      elocal- ; }
vlc-usage(){ cat << EOU

VLC
=====

RHEL/CentOS/SL 7
-------------------

::

    sudo yum install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
        ## already have
    sudo yum install https://download1.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm
    sudo yum install vlc
        ## installed 2.2.8
        ##    46 dependent packages, 
        ##    140MB of libs 
        ##  including ffmpeg-libs and x264-libs 
        ##
        ##  so conflict potential 
        ##   with the manually installed x264- ffmpeg- and obs-    

    #sudo yum install vlc-core                ## (for minimal headless/server install)
    #sudo yum install python-vlc npapi-vlc    ## (optionals)






EOU
}
vlc-dir(){ echo $(local-base)/env/video/vlc/video/vlc-vlc ; }
vlc-cd(){  cd $(vlc-dir); }
vlc-mate(){ mate $(vlc-dir) ; }
vlc-get(){
   local dir=$(dirname $(vlc-dir)) &&  mkdir -p $dir && cd $dir

}
