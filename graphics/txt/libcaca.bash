# === func-gen- : graphics/txt/libcaca fgp graphics/txt/libcaca.bash fgn libcaca fgh graphics/txt
libcaca-src(){      echo graphics/txt/libcaca.bash ; }
libcaca-source(){   echo ${BASH_SOURCE:-$(env-home)/$(libcaca-src)} ; }
libcaca-vi(){       vi $(libcaca-source) ; }
libcaca-env(){      elocal- ; }
libcaca-usage(){ cat << EOU



libcaca is a graphics library that outputs text instead of pixels, so that it
can work on older video cards or text terminals. It is not unlike the famous AAlib library

* http://caca.zoy.org/wiki/libcaca

* http://people.zoy.org/~sam/libcaca/doc/libcaca-tutorial.html

* http://aa-project.sourceforge.net/aalib/

* https://github.com/cacalabs/libcaca


EOU
}
libcaca-dir(){ echo $(local-base)/env/graphics/txt/graphics/txt-libcaca ; }
libcaca-cd(){  cd $(libcaca-dir); }
libcaca-mate(){ mate $(libcaca-dir) ; }
libcaca-get(){
   local dir=$(dirname $(libcaca-dir)) &&  mkdir -p $dir && cd $dir

}
