# === func-gen- : mediamosa/mediamosa fgp mediamosa/mediamosa.bash fgn mediamosa fgh mediamosa
mediamosa-src(){      echo mediamosa/mediamosa.bash ; }
mediamosa-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mediamosa-src)} ; }
mediamosa-vi(){       vi $(mediamosa-source) ; }
mediamosa-env(){      elocal- ; }
mediamosa-usage(){
  cat << EOU
     mediamosa-src : $(mediamosa-src)
     mediamosa-dir : $(mediamosa-dir)

     http://www.mediamosa.org/trac/wiki


EOU
}
mediamosa-dir(){ echo $(local-base)/env/mediamosa/mediamosa-mediamosa ; }
mediamosa-cd(){  cd $(mediamosa-dir); }
mediamosa-mate(){ mate $(mediamosa-dir) ; }
mediamosa-get(){
   local dir=$(dirname $(mediamosa-dir)) &&  mkdir -p $dir && cd $dir

}
