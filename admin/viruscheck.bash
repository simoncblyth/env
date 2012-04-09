# === func-gen- : admin/viruscheck fgp admin/viruscheck.bash fgn viruscheck fgh admin
viruscheck-src(){      echo admin/viruscheck.bash ; }
viruscheck-source(){   echo ${BASH_SOURCE:-$(env-home)/$(viruscheck-src)} ; }
viruscheck-vi(){       vi $(viruscheck-source) ; }
viruscheck-env(){      elocal- ; }
viruscheck-usage(){
  cat << EOU
     viruscheck-src : $(viruscheck-src)
     viruscheck-dir : $(viruscheck-dir)


     http://arstechnica.com/apple/news/2012/04/how-to-check-forand-get-rid-ofa-mac-flashback-infection.ars

EOU
}
viruscheck-dir(){ echo $(local-base)/env/admin/admin-viruscheck ; }
viruscheck-cd(){  cd $(viruscheck-dir); }
viruscheck-mate(){ mate $(viruscheck-dir) ; }
viruscheck-get(){
   local dir=$(dirname $(viruscheck-dir)) &&  mkdir -p $dir && cd $dir

}

viruscheck(){

   defaults read ~/.MacOSX/environment DYLD_INSERT_LIBRARIES

   defaults read /Applications/Safari.app/Contents/Info LSEnvironment

   defaults read /Applications/Firefox.app/Contents/Info LSEnvironment

   env | grep DYLD_
}

