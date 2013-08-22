# === func-gen- : muon_simulation/chroma/chroma fgp muon_simulation/chroma/chroma.bash fgn chroma fgh muon_simulation/chroma
chroma-src(){      echo muon_simulation/chroma/chroma.bash ; }
chroma-source(){   echo ${BASH_SOURCE:-$(env-home)/$(chroma-src)} ; }
chroma-vi(){       vi $(chroma-source) ; }
chroma-env(){      elocal- ; }
chroma-usage(){ cat << EOU





EOU
}
chroma-dir(){ echo $(local-base)/env/muon_simulation/chroma/muon_simulation/chroma ; }
chroma-cd(){  cd $(chroma-dir); }
chroma-mate(){ mate $(chroma-dir) ; }
chroma-get(){
   local dir=$(dirname $(chroma-dir)) &&  mkdir -p $dir && cd $dir

   hg clone https://bitbucket.org/chroma/chroma

}
