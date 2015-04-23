# === func-gen- : tools/word/word fgp tools/word/word.bash fgn word fgh tools/word
word-src(){      echo tools/word/word.bash ; }
word-source(){   echo ${BASH_SOURCE:-$(env-home)/$(word-src)} ; }
word-vi(){       vi $(word-source) ; }
word-env(){      elocal- ; }
word-usage(){ cat << EOU





EOU
}
word-dir(){ echo $(local-base)/env/tools/word/tools/word-word ; }
word-cd(){  cd $(word-dir); }
word-mate(){ mate $(word-dir) ; }
word-get(){
   local dir=$(dirname $(word-dir)) &&  mkdir -p $dir && cd $dir

}
