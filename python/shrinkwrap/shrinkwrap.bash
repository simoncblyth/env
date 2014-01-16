# === func-gen- : python/shrinkwrap/shrinkwrap fgp python/shrinkwrap/shrinkwrap.bash fgn shrinkwrap fgh python/shrinkwrap
shrinkwrap-src(){      echo python/shrinkwrap/shrinkwrap.bash ; }
shrinkwrap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(shrinkwrap-src)} ; }
shrinkwrap-vi(){       vi $(shrinkwrap-source) ; }
shrinkwrap-env(){      elocal- ; }
shrinkwrap-usage(){ cat << EOU

SHRINKWRAP
==========

Python packaging for everything 

* https://shrinkwrap.readthedocs.org/en/latest/index.html

::

    hg clone http://bitbucket.org/seibert/shrinkwrap


FUNCTIONS
----------

shrinkwrap-dupe
    recursively duplicate a shrinkwrap package repository 


EOU
}
shrinkwrap-dir(){ echo $(local-base)/env/python/shrinkwrap/python/shrinkwrap-shrinkwrap ; }
shrinkwrap-cd(){  cd $(shrinkwrap-dir); }
shrinkwrap-mate(){ mate $(shrinkwrap-dir) ; }
shrinkwrap-get(){
   local dir=$(dirname $(shrinkwrap-dir)) &&  mkdir -p $dir && cd $dir

}

shrinkwrap-links(){
   local url=$1
   curl -s $url | perl -ne 'm,href="(\w\S*)", && print "$1\n"' -   
   # skips links starting with non-word char like "?"
}
shrinkwrap-dupe-(){
   local url=$1
   local rel
   shrinkwrap-visit $url
   shrinkwrap-links $url | while read rel ; do
        shrinkwrap-dupe- $url$rel   
        # assumes sensible relative links
   done
}
shrinkwrap-root(){
   echo ${SHRINKWRAP_ROOT:-http://mtrr.org/chroma_pkgs}/
}
shrinkwrap-base(){
   echo ${SHRINKWRAP_BASE:-/tmp/env/chroma_pkgs}/
}
shrinkwrap-dupe(){
   $FUNCNAME- $(shrinkwrap-root)
}
shrinkwrap-visit(){
    local url=$1
    local name=$(basename $url)
    local path=$(shrinkwrap-path $url)
    local fmt="%5s %35s :  %50s %s\n"
    case $name in 
       *.tar.gz) printf "$fmt" TGZ $name $path $url && shrinkwrap-download $url $path ;; 
              *) printf "$fmt" -   $name $path $url ;;
    esac
}
shrinkwrap-download(){
    local url=$1
    local path=$2
    case $path in 
       *.tar.gz) shrinkwrap-download- $* ;;
    esac
}
shrinkwrap-download-(){
    local url=$1
    local path=$2
    [ -f $path ] && echo $msg already downloaded $path && return 0
    local dir=$(dirname $path)
    mkdir -p $dir
    local cmd="curl -s -o $path $url "
    echo $msg $cmd
    eval $cmd
}
shrinkwrap-relative(){
    local url=$1
    local root=$(shrinkwrap-root)
    local rurl=${url/$root}
    echo ${rurl:-DUMMY}
}
shrinkwrap-path(){
    local url=$1
    local rurl=$(shrinkwrap-relative $url)
    case $rurl in
       DUMMY) echo DUMMY ;;
           *) echo $(shrinkwrap-base)$rurl ;;
    esac
}


