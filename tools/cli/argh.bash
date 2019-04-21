# === func-gen- : tools/cli/argh fgp tools/cli/argh.bash fgn argh fgh tools/cli src base/func.bash
argh-source(){   echo ${BASH_SOURCE} ; }
argh-edir(){ echo $(dirname $(argh-source)) ; }
argh-ecd(){  cd $(argh-edir); }
argh-dir(){  echo $LOCAL_BASE/env/tools/cli/argh ; }
argh-cd(){   cd $(argh-dir); }
argh-vi(){   vi $(argh-source) ; }
argh-env(){  elocal- ; }
argh-usage(){ cat << EOU





EOU
}
argh-get(){
   local dir=$(dirname $(argh-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d argh ] && git clone git@github.com:simoncblyth/argh.git
}

argh-h(){ echo $(argh-dir)/argh.h ; }




