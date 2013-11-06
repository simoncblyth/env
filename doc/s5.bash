# === func-gen- : doc/s5 fgp doc/s5.bash fgn s5 fgh doc
s5-src(){      echo doc/s5.bash ; }
s5-source(){   echo ${BASH_SOURCE:-$(env-home)/$(s5-src)} ; }
s5-vi(){       vi $(s5-source) ; }
s5-env(){      elocal- ; }
s5-usage(){ cat << EOU

S5 
===

* http://meyerweb.com/eric/tools/s5/




EOU
}
s5-dir(){ echo $(local-base)/env/doc/s5 ; }
s5-cd(){  cd $(s5-dir); }
s5-mate(){ mate $(s5-dir) ; }
s5-get(){
   local dir=$(dirname $(s5-dir)) &&  mkdir -p $dir && cd $dir

   local url=http://meyerweb.com/eric/tools/s5/s5-blank.zip
   local zip=$(basename $url)

   [ ! -f "$zip" ] && curl -O $url


}
