# === func-gen- : nuwa/cq/cq fgp nuwa/cq/cq.bash fgn cq fgh nuwa/cq
cq-src(){      echo nuwa/cq/cq.bash ; }
cq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cq-src)} ; }
cq-vi(){       vi $(cq-source) ; }
cq-env(){      elocal- ; }
cq-usage(){ cat << EOU





EOU
}
cq-dir(){ echo $(local-base)/env/nuwa/CQ ; }
cq-cd(){  cd $(cq-dir); }
cq-mate(){ mate $(cq-dir) ; }
cq-get(){
   local dir=$(dirname $(cq-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d CQ ] && svn co http://dayabay.ihep.ac.cn/svn/dybsvn/dybgaudi/trunk/DataQuality/DQDump/share/CQ

}
