# === func-gen- : git/gitweb fgp git/gitweb.bash fgn gitweb fgh git
gitweb-src(){      echo git/gitweb.bash ; }
gitweb-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gitweb-src)} ; }
gitweb-vi(){       vi $(gitweb-source) ; }
gitweb-env(){      elocal- ; }
gitweb-usage(){
  cat << EOU
     gitweb-src : $(gitweb-src)
     gitweb-dir : $(gitweb-dir)

     http://git.or.cz/gitwiki/Gitweb
    



EOU
}
gitweb-dir(){ echo $(local-base)/env/git/git-gitweb ; }
gitweb-cd(){  cd $(gitweb-dir); }
gitweb-mate(){ mate $(gitweb-dir) ; }
gitweb-get(){
   local dir=$(dirname $(gitweb-dir)) &&  mkdir -p $dir && cd $dir

}
