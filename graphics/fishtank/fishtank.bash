# === func-gen- : graphics/fishtank/fishtank fgp graphics/fishtank/fishtank.bash fgn fishtank fgh graphics/fishtank
fishtank-src(){      echo graphics/fishtank/fishtank.bash ; }
fishtank-source(){   echo ${BASH_SOURCE:-$(env-home)/$(fishtank-src)} ; }
fishtank-vi(){       vi $(fishtank-source) ; }
fishtank-env(){      elocal- ; }
fishtank-usage(){ cat << EOU





EOU
}
fishtank-dir(){ echo $(local-base)/env/graphics/fishtank; }
fishtank-cd(){  cd $(fishtank-dir); }
fishtank-mate(){ mate $(fishtank-dir) ; }
fishtank-get(){
   local dir=$(dirname $(fishtank-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/muggenhor/fishtank.git

}
