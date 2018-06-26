# === func-gen- : hg/mercurial fgp hg/mercurial.bash fgn mercurial fgh hg
mercurial-src(){      echo hg/mercurial.bash ; }
mercurial-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mercurial-src)} ; }
mercurial-vi(){       vi $(mercurial-source) ; }
mercurial-env(){      elocal- ; }
mercurial-usage(){ cat << EOU

Mercurial Installation
=======================

Get and build mercurial locally, see hg-vi for usage tips/refs.

::

    mercurial-
    mercurial-get
    mercurial-make local

    mercurial-fn  
       # copy and paste the function into .bash_profile    


EOU
}
mercurial-dir(){ echo $(local-base)/env/hg/$(mercurial-name) ; }
mercurial-cd(){  cd $(mercurial-dir); }
mercurial-mate(){ mate $(mercurial-dir) ; }
mercurial-name(){ echo mercurial-3.2 ; }
mercurial-url(){ echo http://mercurial.selenic.com/release/$(mercurial-name).tar.gz ; }

mercurial-get(){
   local dir=$(dirname $(mercurial-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(mercurial-url)
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}

   [ ! -f "$tgz" ] && curl -L -O $url 
   [ ! -d "$nam" ] && tar zxvf $tgz
}

mercurial-make(){
   mercurial-cd
   make $*
}

mercurial-fn(){ cat << EOF
hgl(){ $(mercurial-dir)/hg \$* ; }
EOF
}

